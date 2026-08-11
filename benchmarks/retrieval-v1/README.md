# Generic retrieval benchmark v1

This small, project-authored corpus provides a reproducible lexical baseline before
CorpusDock selects a local embedding model. It contains no user documents and is not
derived from the private development corpus.

The five cases cover exact lexical retrieval, phrase retrieval, cross-source recall,
locator resolution, and paraphrase retrieval. With the bundled sources chunked by the
deterministic rule processor, the expected SQLite FTS5 baseline at limit 10 is:

| Metric | Expected value |
|---|---:|
| Micro source Recall@10 | `0.833333` |
| MRR@10 | `0.8` |
| Locator accuracy | `0.833333` |
| Source verification rate | `1.0` |
| Paraphrase Recall@10 | `0.0` |

The zero paraphrase score is intentional. A local semantic or hybrid retriever must
improve it without reducing locator accuracy or source verification.

## Run the baseline

```bash
benchmark_project="$(mktemp -d)"
corpusdock init "$benchmark_project"
corpusdock ingest benchmarks/retrieval-v1/corpus \
  --project "$benchmark_project" --sentence-processor rule
corpusdock index --project "$benchmark_project"
corpusdock eval benchmarks/retrieval-v1/judgments.json \
  --project "$benchmark_project" --json
```

Use an equivalent temporary directory on platforms without `mktemp`. The original
benchmark files remain authoritative; the temporary manifest, artifacts, chunks, and
index are disposable.

## Dataset schema

`judgments.json` uses schema version 1:

```json
{
  "schema_version": 1,
  "name": "corpusdock-generic-retrieval-v1",
  "description": "A human-readable description.",
  "cases": [
    {
      "case_id": "stable-lowercase-id",
      "category": "lexical",
      "query": "literal query text",
      "match_mode": "all",
      "relevance": [
        {
          "source_id": "src-<sha256>",
          "locator": {
            "locator_type": "text_line",
            "line_start": 2,
            "line_end": 2
          }
        }
      ]
    }
  ]
}
```

`match_mode` is `all`, `any`, or `phrase`. Each case requires at least one relevant
content-derived source ID. A locator is optional; when present, every supplied field
must match one of the exact stored locators returned with a relevant result.

Reports record the dataset SHA-256, CorpusDock version, index build metadata, index
size, search-only latency, and process peak RSS when supported. They intentionally
omit excerpts, formatted citations, and local source paths. Peak RSS is a process
high-water mark rather than an isolated incremental allocation measurement.
