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
search, and deterministic reciprocal-rank fusion combines lexical and semantic
candidates without changing their evidence. An optional local structured-extraction
stage can derive reviewable concept mentions, claims, stance, and relations from
those exact evidence chunks. Derived analysis never replaces source evidence.

Format extraction does not invoke Calibre, LibreOffice, Pandoc, or another system
converter. TXT, EPUB, DOCX, and unencrypted MOBI parsing is implemented in the
CorpusDock package using Python's standard library; MOBI includes uncompressed,
PalmDOC, and HUFF/CDIC decoding. PDF text-layer extraction uses the pinned `pypdf`
package. DRM-protected MOBI files and unknown compression types fail explicitly.

## Install and ingest

```bash
uv sync --extra local-models
corpusdock init .
corpusdock sync ./documents
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

`ingest` is additive: it never removes an earlier source merely because a path is
absent from a later command. For a long-running corpus, use an authoritative input
mirror instead:

```bash
corpusdock sync ./documents
corpusdock sync ./documents --configure-only
```

`sync` recursively hashes the directory, makes the manifest match its current
contents, processes only missing or stale extraction/chunk artifacts, and atomically
refreshes the exact index. Identical bytes have one content-derived source ID, so a
rename updates path provenance without extracting or chunking the book again. New or
changed bytes enter the queue; removing the last copy of a source from the directory
prunes its extraction, chunks, semantic vector cache, and derived analysis records.
An empty configured directory intentionally produces an empty corpus.

The configuration is stored in ignored `.corpusdock/pipeline.json`. After either
`sync` form configures an input, every `corpusdock analyze` resume scans that input
before loading the local AI model:

```bash
corpusdock sync ./documents --configure-only
corpusdock analyze --analysis-runtime vllm --device cuda --dtype bfloat16
```

The manifest, each per-source extraction, each per-source chunk artifact, the exact
index, and each analysis batch are durable atomic checkpoints. A hard interruption
may retry only the currently uncommitted unit; it cannot create duplicate committed
sources, chunks, evidence IDs, vectors, or analysis rows.

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
subdirectory, `ingest`, `sync`, `index`, `embed`, `analyze`, `search`, `eval`, `verify`,
`source`, and `doctor` discover the nearest initialized project; use
`--project /path/to/project` to select one explicitly.

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

### Persistent local semantic and hybrid search

Install the optional semantic runtime, build vectors from the exact index, and query
them locally:

```bash
uv pip install --torch-backend cpu -e '.[semantic]'
corpusdock embed --allow-model-download --device cpu
corpusdock search "how teams preserve operational knowledge" \
  --retrieval semantic --device cpu --json
corpusdock search "how teams preserve operational knowledge" \
  --retrieval hybrid --device cpu --json
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
whose content changed. The builder reuses vectors by stable evidence ID and embeds
only new or changed chunks; removed evidence is omitted. Semantic search rejects a
stale or checksum-invalid index until that atomic rebuild completes.

Hybrid search requests up to 60 candidates from both SQLite FTS5 and the persistent
semantic index, then applies equal-weight reciprocal-rank fusion with `k=60`.
Agreement between the independent rankings is rewarded without trying to normalize
incompatible BM25 and cosine scores. Ties use lexical rank, semantic rank, and stable
evidence ID in that order, so the same indexes produce the same result order across
platforms. The returned score is the fusion score; excerpts and citations still come
unchanged from the exact index. Use `--match any` when a long natural-language query
should contribute broad lexical candidates.

On supported NVIDIA hardware, install the CUDA runtime and use the GPU for both the
one-time corpus build and query inference:

```bash
uv pip install --torch-backend auto -e '.[semantic]'
corpusdock embed --allow-model-download --device cuda
corpusdock search "how teams preserve operational knowledge" \
  --retrieval hybrid --device cuda
```

### Evidence-grounded local analysis

CorpusDock can turn exact retrieval chunks into reviewable concept mentions, claims,
stance, and relations without sending passages to a hosted model. The portable
Transformers runtime works on CPU or CUDA; the optional vLLM runtime provides the
selected high-throughput NVIDIA path.

Install vLLM and first run the public regression gate:

```bash
uv sync --extra analysis-vllm
corpusdock analysis-eval benchmarks/analysis-v1/cases.json \
  --analysis-runtime vllm --allow-model-download \
  --device cuda --dtype bfloat16 --json
```

The measured default is `Qwen/Qwen3.5-4B` in BF16. CorpusDock resolves an immutable
model revision, requires safetensors-only weights, rejects repositories containing
Python or `auto_map` hooks, loads with `trust_remote_code=False`, disables thinking
and stochastic sampling, and validates every candidate against exact local evidence.
The first explicit download fetches public model files only. When downloads are not
explicitly allowed, CorpusDock forces the registry offline and disables vLLM usage
telemetry.

After the public gate passes, run a bounded pilot and then start full-corpus
analysis:

```bash
corpusdock analyze --analysis-runtime vllm --device cuda \
  --dtype bfloat16 --limit 12 --json
corpusdock analyze --analysis-runtime vllm --device cuda \
  --dtype bfloat16 --json
corpusdock doctor --json
```

The pilot and full-corpus scopes are distinct runs. Repeating either command resumes
when its selection, prompt, and stable extractor configuration match. Stable evidence
that remains in the corpus is reused even when books were added, removed, changed, or
moved between resumes; only newly selected evidence is sent through the local model.
Each completed batch is committed atomically. Change `--no-resume` to start a distinct
run. `corpusdock doctor --json` reports the latest run's exact scope, status, and
committed progress without exposing document content. vLLM defaults to a
cache-safe maximum batch of 16. Larger GPUs can opt into 32, while smaller GPUs can
reduce `--analysis-batch-size` or `--vllm-gpu-memory-utilization`. The selected RTX
5080 public profile processed its nine requests in one 17.23-second batch while
preserving a `1.0` valid and fully grounded response rate.

Analysis lives in ignored `.corpusdock/analysis.sqlite3`. It stores candidate labels,
standalone propositions, typed relations, review state, evidence IDs, chunk-relative
support offsets, and support hashes. The model cites local SaT evidence-unit IDs;
deterministic code converts them to continuous exact spans. The database intentionally
stores no source paths, excerpts, raw model output, or prompts. Claims preserve
polarity, certainty, conditionality, attribution, and normative force so disagreement
is retained rather than silently reconciled. Candidate IDs and anchors prepare the
next phase—cross-evidence concept resolution and graph querying—but candidates are
not accepted facts until reviewed.

For the portable fallback, install `.[analysis]` and omit
`--analysis-runtime vllm`. Transformers also supports the optional
`.[analysis,analysis-cuda]` bitsandbytes bakeoff path. Neither dynamic vLLM FP8 nor a
larger 9B NF4 model beat the selected BF16 profile on the versioned quality gate, so
faster or larger configurations are not selected at the expense of fidelity. See the
[analysis benchmark and model decision](benchmarks/analysis-v1/README.md) for exact
revisions, gates, measurements, limitations, and reproduction commands.

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
`--retrieval semantic` or `--retrieval hybrid`. After building the persistent index,
evaluate the deployed fused path with:

```bash
corpusdock eval benchmarks/retrieval-v2/judgments.json \
  --project "$semantic_project" --limit 3 --retrieval hybrid \
  --device cuda --no-verify --json
```

See the
[multilingual benchmark and model decision](benchmarks/retrieval-v2/README.md) for
exact revisions, CPU/RAM/VRAM measurements, hybrid quality gates, rejected reranker
experiment, limitations, and the reproducible lexical control.

## Principles

- Local-first: no source content leaves the machine by default.
- Citation-first: every retrieval result carries a durable source locator.
- Vendor-neutral: the core does not require an OpenAI, Anthropic, or other API key.
- Auditable: preserve originals, hashes, extraction records, and evidence anchors.
