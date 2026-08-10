"""Dependency-free MOBI HUFF/CDIC decompression.

The decoder is shipped as part of CorpusDock and operates only on unencrypted
records supplied by the MOBI parser.  It does not invoke a converter executable,
load a shared library, access the network, or implement DRM removal.
"""

from __future__ import annotations

import struct
from typing import Final


HUFF_MAGIC: Final = b"HUFF\x00\x00\x00\x18"
CDIC_MAGIC: Final = b"CDIC\x00\x00\x00\x10"
_MAX_CODE_LENGTH: Final = 32
_MAX_DICTIONARY_BITS: Final = 16
_MAX_RECURSION_DEPTH: Final = 64
_MAX_CACHE_BYTES: Final = 256 * 1024 * 1024


class HuffCdicError(ValueError):
    """A malformed or unsafe HUFF/CDIC stream."""


class HuffCdicDecoder:
    """Decode MOBI compression type ``0x4448`` without external dependencies."""

    def __init__(
        self,
        huff_record: bytes,
        cdic_records: tuple[bytes, ...],
        *,
        max_output_bytes: int,
    ) -> None:
        if max_output_bytes <= 0:
            raise HuffCdicError("HUFF/CDIC output limit must be positive.")
        self._max_output_bytes = max_output_bytes
        self._max_cache_bytes = min(
            _MAX_CACHE_BYTES,
            max(1024 * 1024, max_output_bytes * 4),
        )
        self._cached_bytes = 0
        self._primary, self._minimum_codes, self._maximum_codes = (
            self._read_huff_record(huff_record)
        )
        self._dictionary = self._read_cdic_records(cdic_records)
        self._expanded: list[bytes | None] = [None] * len(self._dictionary)

    def decode_records(self, records: tuple[bytes, ...]) -> bytes:
        """Decode text records while enforcing one aggregate output bound."""

        output = bytearray()
        for record in records:
            remaining = self._max_output_bytes - len(output)
            if remaining <= 0 and record:
                raise HuffCdicError("HUFF/CDIC output exceeds the declared text size.")
            output.extend(self._decode_stream(record, remaining, ()))
        return bytes(output)

    def _decode_stream(
        self,
        payload: bytes,
        output_limit: int,
        expansion_stack: tuple[int, ...],
    ) -> bytes:
        if not payload:
            return b""

        stream = int.from_bytes(payload, "big")
        bit_count = len(payload) * 8
        bit_position = 0
        output = bytearray()
        while bit_position < bit_count:
            remaining_bits = bit_count - bit_position
            if remaining_bits >= _MAX_CODE_LENGTH:
                code = (stream >> (remaining_bits - _MAX_CODE_LENGTH)) & 0xFFFFFFFF
            else:
                code = (stream & ((1 << remaining_bits) - 1)) << (
                    _MAX_CODE_LENGTH - remaining_bits
                )

            code_length, terminal, maximum_code = self._primary[code >> 24]
            if not terminal:
                while (
                    code_length <= _MAX_CODE_LENGTH
                    and code < self._minimum_codes[code_length]
                ):
                    code_length += 1
                if code_length > _MAX_CODE_LENGTH:
                    raise HuffCdicError("HUFF code does not match the decoding table.")
                maximum_code = self._maximum_codes[code_length]

            # MOBI records may end with padding shorter than the selected code.
            if code_length > remaining_bits:
                break
            if maximum_code < code:
                raise HuffCdicError("HUFF code lies above its decoding range.")

            dictionary_index = (maximum_code - code) >> (_MAX_CODE_LENGTH - code_length)
            if dictionary_index >= len(self._dictionary):
                raise HuffCdicError("HUFF code references a missing CDIC phrase.")

            phrase = self._expand_phrase(dictionary_index, expansion_stack)
            if len(output) + len(phrase) > output_limit:
                raise HuffCdicError("HUFF/CDIC output exceeds the declared text size.")
            output.extend(phrase)
            bit_position += code_length
        return bytes(output)

    def _expand_phrase(
        self, dictionary_index: int, expansion_stack: tuple[int, ...]
    ) -> bytes:
        payload, is_literal = self._dictionary[dictionary_index]
        if is_literal:
            return payload

        cached = self._expanded[dictionary_index]
        if cached is not None:
            return cached
        if dictionary_index in expansion_stack:
            raise HuffCdicError("CDIC dictionary contains a recursive phrase cycle.")
        if len(expansion_stack) >= _MAX_RECURSION_DEPTH:
            raise HuffCdicError("CDIC phrase nesting exceeds the safe depth limit.")

        expanded = self._decode_stream(
            payload,
            self._max_output_bytes,
            (*expansion_stack, dictionary_index),
        )
        if self._cached_bytes + len(expanded) > self._max_cache_bytes:
            raise HuffCdicError(
                "Expanded CDIC dictionary exceeds the safe cache limit."
            )
        self._expanded[dictionary_index] = expanded
        self._cached_bytes += len(expanded)
        return expanded

    @staticmethod
    def _read_huff_record(
        payload: bytes,
    ) -> tuple[
        tuple[tuple[int, bool, int], ...],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        if len(payload) < 24 or payload[:8] != HUFF_MAGIC:
            raise HuffCdicError("MOBI HUFF record header is invalid.")
        primary_offset, secondary_offset = struct.unpack_from(">II", payload, 8)
        primary_values = _read_uint32_table(
            payload, primary_offset, 256, "HUFF primary"
        )
        secondary_values = _read_uint32_table(
            payload, secondary_offset, 64, "HUFF secondary"
        )

        primary: list[tuple[int, bool, int]] = []
        for value in primary_values:
            code_length = value & 0x1F
            terminal = bool(value & 0x80)
            if not 1 <= code_length <= _MAX_CODE_LENGTH:
                raise HuffCdicError("HUFF primary table has an invalid code length.")
            if code_length <= 8 and not terminal:
                raise HuffCdicError("HUFF primary table has an incomplete short code.")
            maximum_code = (((value >> 8) + 1) << (_MAX_CODE_LENGTH - code_length)) - 1
            primary.append((code_length, terminal, maximum_code))

        minimum_codes = [0] * (_MAX_CODE_LENGTH + 1)
        maximum_codes = [0] * (_MAX_CODE_LENGTH + 1)
        for code_length in range(1, _MAX_CODE_LENGTH + 1):
            shift = _MAX_CODE_LENGTH - code_length
            minimum_codes[code_length] = (
                secondary_values[(code_length - 1) * 2] << shift
            )
            maximum_codes[code_length] = (
                (secondary_values[(code_length - 1) * 2 + 1] + 1) << shift
            ) - 1
        return tuple(primary), tuple(minimum_codes), tuple(maximum_codes)

    @staticmethod
    def _read_cdic_records(
        records: tuple[bytes, ...],
    ) -> tuple[tuple[bytes, bool], ...]:
        if not records:
            raise HuffCdicError("MOBI HUFF data has no CDIC records.")

        dictionary: list[tuple[bytes, bool]] = []
        expected_phrases: int | None = None
        expected_bits: int | None = None
        for payload in records:
            if len(payload) < 16 or payload[:8] != CDIC_MAGIC:
                raise HuffCdicError("MOBI CDIC record header is invalid.")
            phrase_count, dictionary_bits = struct.unpack_from(">II", payload, 8)
            if phrase_count <= 0:
                raise HuffCdicError("CDIC dictionary contains no phrases.")
            if dictionary_bits > _MAX_DICTIONARY_BITS:
                raise HuffCdicError("CDIC dictionary bit width is unsafe or invalid.")
            if expected_phrases is None:
                expected_phrases = phrase_count
                expected_bits = dictionary_bits
            elif phrase_count != expected_phrases or dictionary_bits != expected_bits:
                raise HuffCdicError(
                    "CDIC records disagree about dictionary dimensions."
                )

            remaining = phrase_count - len(dictionary)
            if remaining <= 0:
                raise HuffCdicError(
                    "CDIC records contain entries beyond the dictionary."
                )
            entries_in_record = min(1 << dictionary_bits, remaining)
            offset_table_end = 16 + entries_in_record * 2
            if offset_table_end > len(payload):
                raise HuffCdicError("CDIC phrase-offset table is truncated.")
            offsets = struct.unpack_from(f">{entries_in_record}H", payload, 16)
            for offset in offsets:
                entry_start = 16 + offset
                if entry_start + 2 > len(payload):
                    raise HuffCdicError("CDIC phrase offset lies outside its record.")
                length_and_flag = struct.unpack_from(">H", payload, entry_start)[0]
                phrase_length = length_and_flag & 0x7FFF
                phrase_start = entry_start + 2
                phrase_end = phrase_start + phrase_length
                if phrase_end > len(payload):
                    raise HuffCdicError("CDIC phrase data is truncated.")
                dictionary.append(
                    (payload[phrase_start:phrase_end], bool(length_and_flag & 0x8000))
                )

        assert expected_phrases is not None
        if len(dictionary) != expected_phrases:
            raise HuffCdicError("CDIC dictionary is incomplete.")
        return tuple(dictionary)


def _read_uint32_table(
    payload: bytes, offset: int, count: int, table_name: str
) -> tuple[int, ...]:
    if offset < 24 or offset + count * 4 > len(payload):
        raise HuffCdicError(f"{table_name} table lies outside the HUFF record.")
    return struct.unpack_from(f">{count}I", payload, offset)
