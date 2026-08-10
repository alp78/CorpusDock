# Local data directory

This directory documents local-only data locations. Do not commit source documents,
extracted text, databases, embeddings, or model weights.

Suggested local layout:

```text
.corpusdock/
  manifest.json  # Source registration manifest
  extracted/     # Exact parser output and source anchors
  chunks/        # Exact anchor-aware retrieval chunks
  index.sqlite3  # Derived embedded SQLite FTS5 index
data/
  originals/     # Optional immutable user source files
  indexes/       # Reserved for future external/vector indexes
```

The root `.gitignore` excludes the manifest and all content-bearing paths by default.
