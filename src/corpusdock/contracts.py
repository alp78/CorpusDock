"""Stable, provider-neutral data contracts for CorpusDock results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CitationLocator:
    """A durable pointer back to a source location."""

    source_id: str
    locator_type: str
    label: str
    page: int | None = None
    page_label: str | None = None
    chapter: str | None = None
    heading: str | None = None
    spine_item: str | None = None
    paragraph_id: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    extraction_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    """Extraction coverage that qualifies what a retrieved excerpt represents."""

    extraction_status: str
    unresolved_pdf_pages: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction_status": self.extraction_status,
            "unresolved_pdf_pages": list(self.unresolved_pdf_pages),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    """A search hit that an agent can cite and a human can verify."""

    evidence_id: str
    excerpt: str
    citation: str
    locator: CitationLocator
    source_path: str
    verification_status: str
    score: float | None = None
    chunk_id: str | None = None
    anchor_ids: tuple[str, ...] = ()
    locators: tuple[CitationLocator, ...] = ()
    source_coverage: SourceCoverage | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["locator"] = self.locator.to_dict()
        result["anchor_ids"] = list(self.anchor_ids)
        result["locators"] = [locator.to_dict() for locator in self.locators]
        result["source_coverage"] = (
            self.source_coverage.to_dict() if self.source_coverage is not None else None
        )
        return result
