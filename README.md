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
unresolved and are never inferred from images. A versioned evaluator compares exact
search with provider-neutral, local semantic retrieval while preserving the identical
evidence and citation contract. A persistent local vector index now powers semantic
search; hybrid ranking is the next retrieval stage.

Format extraction does not invoke Calibre, LibreOffice, Pandoc, or another system
converter. TXT, EPUB, DOCX, and unencrypted MOBI parsing is implemented in the
CorpusDock package using Python's standard library; MOBI includes uncompressed,
PalmDOC, and HUFF/CDIC decoding. PDF text-layer extraction uses the pinned `pypdf`
package. DRM-protected MOBI files and unknown compression types fail explicitly.

## Install and ingest

```bash
uv sync --extra local-models
corpusdock init .
corpusdock ingest ./documents
corpusdock index
corpusdock doctor
corpusdock search "citation anchors" --json
corpusdock eval ./judgments.json --json
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
subdirectory, `ingest`, `index`, `embed`, `search`, `eval`, `verify`, `source`, and
`doctor` discover the nearest initialized project; use `--project /path/to/project`
to select one explicitly.

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

### Persistent local semantic search

Install the optional semantic runtime, build vectors from the exact index, and query
them locally:

```bash
uv pip install --torch-backend cpu -e '.[semantic]'
corpusdock embed --allow-model-download --device cpu
corpusdock search "how teams preserve operational knowledge" \
  --retrieval semantic --device cpu --json
```

`--allow-model-download` is needed only for the explicit first fetch of public model
files. It never permits uploading document text. The resolved immutable revision is
recorded in `.corpusdock/semantic.sqlite3`; later `embed` and semantic `search` runs
are local-only by default and reuse `.corpusdock/models/`. Pass a compatible local
model directory with `--embedding-model /path/to/model` to avoid a registry fetch
entirely.

The semantic database contains normalized float32 vectors and stable source, chunk,
and evidence IDs, but no excerpts or source paths. Results are resolved from the exact
index and therefore return the same exact excerpts, locators, citations, coverage,
and evidence IDs as lexical search. Run `corpusdock verify <evidence-id>` for live
source-byte verification. Re-run `corpusdock embed` after rebuilding an exact index
whose content changed; semantic search rejects stale or checksum-invalid vectors.

On supported NVIDIA hardware, install the CUDA runtime and use the GPU for both the
one-time corpus build and query inference:

```bash
uv pip install --torch-backend auto -e '.[semantic]'
corpusdock embed --allow-model-download --device cuda
corpusdock search "how teams preserve operational knowledge" \
  --retrieval semantic --device cuda
```

## Retrieval evaluation

`corpusdock eval` measures a local index against a versioned JSON relevance dataset.
It reports micro-averaged source Recall@k, query-averaged MRR@k, locator accuracy,
live source-verification rate, search-only latency, SQLite index size, and process
peak RSS where the operating system exposes it. Reports contain queries and stable
source/evidence IDs, but never retrieved excerpts, citations, or source paths.

The project-authored generic v1 benchmark deliberately includes a paraphrase that
exact lexical search cannot retrieve. Run the reproducible baseline with:

```bash
benchmark_project="$(mktemp -d)"
corpusdock init "$benchmark_project"
corpusdock ingest benchmarks/retrieval-v1/corpus \
  --project "$benchmark_project" --sentence-processor rule
corpusdock index --project "$benchmark_project"
corpusdock eval benchmarks/retrieval-v1/judgments.json \
  --project "$benchmark_project" --json
```

The evaluator rejects datasets whose judged source IDs are absent from the selected
index. Live evidence verification is enabled by default; `--no-verify` measures only
retrieval and locator judgments. See the
[benchmark contract](benchmarks/retrieval-v1/README.md) for the schema and expected
lexical baseline.

### Local semantic evaluation

The v2 benchmark adds generic English paraphrases, same-language multilingual
queries, cross-lingual queries, and overlapping distractors. CorpusDock's measured
quality-first default is `Qwen/Qwen3-Embedding-0.6B` at its full 1024 dimensions.
The semantic adapter is optional, refuses repository Python, resolves an immutable
model revision, and is local-only unless a model download is explicitly permitted.

Install a CPU-only semantic runtime and run the first evaluation:

```bash
uv pip install --torch-backend cpu -e '.[semantic]'
semantic_project="$(mktemp -d)"
corpusdock init "$semantic_project"
corpusdock ingest benchmarks/retrieval-v2/corpus \
  --project "$semantic_project" --sentence-processor rule
corpusdock index --project "$semantic_project"
corpusdock eval benchmarks/retrieval-v2/judgments.json \
  --project "$semantic_project" --limit 3 --retrieval semantic \
  --allow-model-download --device cpu --no-verify --json
```

Only public model files are fetched. CorpusDock does not upload document text; after
the model is cached, omit `--allow-model-download` for a network-free run. A local
model directory can be passed with `--embedding-model /path/to/model`.

On supported NVIDIA hardware, install an automatically selected PyTorch CUDA wheel
and switch inference to the GPU:

```bash
uv pip install --torch-backend auto -e '.[semantic]'
corpusdock eval benchmarks/retrieval-v2/judgments.json \
  --project "$semantic_project" --limit 3 --retrieval semantic \
  --device cuda --no-verify --json
```

Semantic evaluation deliberately builds an ephemeral in-memory matrix so candidate
models and dimensions can be compared without replacing the persistent index.
Production semantic search uses `corpusdock embed` followed by a search with
`--retrieval semantic`. See the
[multilingual benchmark and model decision](benchmarks/retrieval-v2/README.md) for
exact revisions, CPU/RAM/VRAM measurements, CUDA throughput, limitations, and the
reproducible lexical control.

## Principles

- Local-first: no source content leaves the machine by default.
- Citation-first: every retrieval result carries a durable source locator.
- Vendor-neutral: the core does not require an OpenAI, Anthropic, or other API key.
- Auditable: preserve originals, hashes, extraction records, and evidence anchors.
