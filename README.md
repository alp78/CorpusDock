# CorpusDock

**Citation-aware local document ingestion and retrieval for AI agents.**

CorpusDock ingests local documents, preserves their source anchors, and returns exact
full-text evidence that agents can cite and people can verify. It is designed to work with
Codex, Claude Code, and ordinary shells without making any AI provider mandatory.

## Project status

CorpusDock registers, extracts, sentence-chunks, and locally indexes `.txt`, `.pdf`,
`.epub`, `.mobi`, and `.docx` files into versioned local artifacts. Every search hit
returns exact chunk text, stable evidence and chunk IDs, durable source locators, and
extraction coverage. PDF ingestion is text-layer only; scanned pages are reported as
unresolved and are never inferred from images. Semantic retrieval is not implemented
yet.

Format extraction does not invoke Calibre, LibreOffice, Pandoc, or another system
converter. TXT, EPUB, DOCX, and unencrypted MOBI parsing is implemented in the
CorpusDock package using Python's standard library; MOBI includes uncompressed,
PalmDOC, and HUFF/CDIC decoding. PDF text-layer extraction uses the pinned `pypdf`
package. DRM-protected MOBI files and unknown compression types fail explicitly.

## Install and ingest

```bash
uv sync --all-extras
corpusdock init .
corpusdock ingest ./documents
corpusdock index
corpusdock doctor
corpusdock search "citation anchors" --json
```

The default quality profile uses the local ONNX `sat-12l-sm` Segment Any Text model.
The model and tokenizer are downloaded and cached on first use; document text is not
sent with those downloads, and sentence inference runs locally. Once cached, force a
network-isolated run with:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 corpusdock ingest ./documents
```

Use `--sentence-processor rule` for the dependency-free local fallback. It preserves
offsets but is less capable on multilingual or irregular text.

## Ingestion contract

```bash
corpusdock ingest ./documents
corpusdock ingest ./documents --extract-only
corpusdock ingest ./documents --register-only
```

Full ingestion calculates a SHA-256 digest, assigns a content-derived `src-...` ID,
extracts source-anchored text, and creates structure-bounded sentence chunks. Derived
content lives in `.corpusdock/extracted/` and `.corpusdock/chunks/`; both are local and
ignored by Git. The original file remains the source of truth. Re-registering
identical bytes preserves one source ID and records each observed local path.

All parser and decoder code ships with CorpusDock. No external conversion executable
is discovered or called at runtime, and ingestion never sends document content to a
network service.

PDF extraction reads an existing text layer only. A PDF with pages that contain no
embedded text remains `partial`; those pages are recorded as unresolved and cannot
produce fabricated evidence or chunks. Use a separately prepared text-accessible PDF
when scanned-page content is required.

Use `corpusdock source <source-id>` to inspect a registered source as JSON. From a
subdirectory, `ingest`, `index`, `search`, `verify`, `source`, and `doctor` discover
the nearest initialized project; use `--project /path/to/project` to select one
explicitly.

The search database lives at `.corpusdock/index.sqlite3` and is ignored by Git. It is
an atomic, rebuildable SQLite FTS5 index over persisted chunks, not the canonical
corpus. Re-run `corpusdock index` after ingestion. Search refuses stale indexes.

```bash
corpusdock search "data retention policy" --limit 10
corpusdock search '"quality assurance"' --json
corpusdock verify ev-<sha256> --json
```

Search is literal lexical retrieval: `--match all` is the default, with `any` and
`phrase` alternatives. `verify` revalidates the artifact chain and hashes an available
original file before returning `source-anchor-confirmed`.

## Principles

- Local-first: no source content leaves the machine by default.
- Citation-first: every retrieval result carries a durable source locator.
- Vendor-neutral: the core does not require an OpenAI, Anthropic, or other API key.
- Auditable: preserve originals, hashes, extraction records, and evidence anchors.
