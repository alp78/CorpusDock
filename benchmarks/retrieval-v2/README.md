# Multilingual semantic retrieval benchmark v2

This project-authored corpus is a small regression gate for selecting and changing
local embedding models. It contains no user documents. Fifteen generic sources and
29 queries cover exact matches, English paraphrases, same-language retrieval, and
cross-lingual retrieval across ten languages. Deliberate distractors reuse concepts
such as wind, humidity, temperature, bearings, brakes, and preservation.

At limit 3 there are 30 source-and-locator relevance judgments. Every relevant hit
must return the exact stored excerpt and matching text-line locator; semantic scores
never replace CorpusDock's evidence contract.

## Reproducible control

```bash
benchmark_project="$(mktemp -d)"
corpusdock init "$benchmark_project"
corpusdock ingest benchmarks/retrieval-v2/corpus \
  --project "$benchmark_project" --sentence-processor rule
corpusdock index --project "$benchmark_project"
corpusdock eval benchmarks/retrieval-v2/judgments.json \
  --project "$benchmark_project" --limit 3 --no-verify --json
```

The deterministic SQLite FTS5 control retrieves every literal/citation case, but no
paraphrase or multilingual case:

| Metric | Expected value |
|---|---:|
| Micro source Recall@3 | `0.2` |
| MRR@3 | `0.172414` |
| Locator accuracy | `0.2` |
| Paraphrase Recall@3 | `0.0` |
| Same-language multilingual Recall@3 | `0.0` |
| Cross-lingual Recall@3 | `0.0` |

## Local model results

The CPU results below are medians of three clean processes after the selected model
revision was cached. Tests used Python 3.13.13, Sentence Transformers 5.7.0, PyTorch
2.13.0, a Ryzen 7 9800X3D, and batch size 16. Search latency includes one query
embedding and exact cosine ranking over 15 chunks; it excludes model loading,
document embedding, and optional source verification.

| Model and immutable revision | Dimensions | Recall@3 | MRR@3 | Selected model files | Peak RSS | Cached load | Embed 15 docs | Query p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `intfloat/multilingual-e5-small` `614241f6…` | 384 | `0.966667` | `0.931034` | 493 MB | 963 MB | 2,046 ms | 103.8 ms | 17.86 ms |
| `Qwen/Qwen3-Embedding-0.6B` `97b0c614…` | 1024 | `1.0` | `1.0` | 1,207 MB | 1,711 MB | 689 ms | 718.2 ms | 60.10 ms |
| `BAAI/bge-m3` `5617a9f6…` | 1024 | `1.0` | `1.0` | 2,293 MB | 2,116 MB | 1,497 ms | 885.0 ms | 63.05 ms |

E5-small missed one English-to-Czech cross-lingual case. Qwen and BGE-M3 retrieved
all 30 judged sources and locators. Qwen matched BGE-M3's quality here with smaller
model files, lower peak RAM, faster cached loading, and faster document embedding,
so full-width Qwen is CorpusDock's quality-first default. A 256-dimensional Qwen run
missed one source in the multi-source case; dimension truncation is therefore not the
default even though the model supports it.

This corpus is intentionally small. Perfect scores are a regression result, not a
claim that one model is universally best; expand the judgments with representative,
lawfully shareable material before changing the default.

The model cards are the authoritative source for intended usage and licenses:
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
[BGE-M3](https://huggingface.co/BAAI/bge-m3), and
[multilingual-E5-small](https://huggingface.co/intfloat/multilingual-e5-small).
Candidates requiring repository Python through `trust_remote_code` or gated model
access were not eligible for the default.

## Run semantic evaluation

Install the optional runtime with a CPU-only PyTorch wheel:

```bash
uv pip install --torch-backend cpu -e '.[semantic]'
```

The first model fetch is an explicit operation. CorpusDock resolves and reports the
immutable revision, downloads only the tokenizer/configuration and one usable weight
format, refuses repository Python, then performs all document and query inference
locally:

```bash
corpusdock eval benchmarks/retrieval-v2/judgments.json \
  --project "$benchmark_project" --limit 3 --retrieval semantic \
  --allow-model-download --device cpu --no-verify --json
```

After the model is cached, omit `--allow-model-download`. To pin a reproduced run,
pass the complete reported hash with `--model-revision`. Override the default with
`--embedding-model intfloat/multilingual-e5-small` or a compatible local model
directory.

Model registry requests transfer model metadata and weights only. CorpusDock does not
read the benchmark chunks until the model is present, loads the resolved snapshot
with `local_files_only=True`, and never uploads source text.

## Run persistent semantic search

After choosing a model, build an atomic project-local vector index and search it with
the same evidence contract as lexical retrieval:

```bash
corpusdock embed --project "$benchmark_project" \
  --allow-model-download --device cpu --json
corpusdock search "how are cargo records preserved?" \
  --project "$benchmark_project" --retrieval semantic --device cpu --json
```

The first command stores vectors and non-content provenance in the ignored
`.corpusdock/semantic.sqlite3` database. It does not copy excerpts or paths into that
database. Once the model is cached, omit `--allow-model-download`; semantic queries
resolve their hits from the exact index so citations and evidence verification remain
unchanged.

## CUDA

For a supported NVIDIA GPU, install the PyTorch backend selected for the local driver
and request CUDA inference explicitly:

```bash
uv pip install --torch-backend auto -e '.[semantic]'
corpusdock eval benchmarks/retrieval-v2/judgments.json \
  --project "$benchmark_project" --limit 3 --retrieval semantic \
  --device cuda --no-verify --json
```

Use `corpusdock embed --device cuda` and `corpusdock search --retrieval semantic
--device cuda` for the persistent workflow.

On an RTX 5080 with CUDA 13.0, the three-run median Qwen query p50 was 29.48 ms versus
60.10 ms on CPU. The small 15-document build is dominated by CUDA startup, so a
separate throughput check used 1,024 approximately 1.3k-character synthetic documents
made by repeating the bundled project-authored text: CUDA embedded 92.15
documents/second at batch size 16, compared with 2.43 documents/second on CPU. The
CUDA run peaked at 2.67 GB allocated and 3.15 GB reserved
VRAM. Evaluation JSON records the framework, CUDA runtime, GPU name, and peak device
memory for reproducibility.
