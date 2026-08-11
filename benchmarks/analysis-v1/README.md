# Analysis benchmark v1

This project-authored benchmark tests evidence-grounded knowledge extraction without
using private documents. Its nine short, generic passages cover causal and normative
claims, negation, uncertainty, attribution, conditionality, preserved disagreement,
German and Spanish text, an instruction embedded in source data, and abstention on
structural text.

The model returns untrusted candidates. CorpusDock first segments each passage with
the local SaT sentence model and gives the extractor opaque evidence-unit IDs. A
candidate is accepted only when its unit references, IDs, enums, links, bounds, and
stance fields satisfy the versioned contract. CorpusDock derives one exact continuous
support span from those units and persists only evidence-relative offsets and a text
digest—not copied support text or raw output.

Concept matches use their exact support anchor. Claim matches additionally require
the expected claim type, polarity, certainty, conditionality, attribution, and
normative force. Relation matches require the same stance, relation type, grounded
claim, and concept endpoints. `contrasts_with` is the only relation treated as
symmetric. Extra candidates reduce precision because the fixtures are deliberately
short and exhaustively annotated.

[`expected-results.json`](expected-results.json) binds the reviewed baseline to the
SHA-256 digests of both the dataset and static prompt. It records minimum regression
gates, the measured result, runtime observations, and comparison profiles. Evaluation
reports contain case IDs and non-content provenance but never fixture passages or raw
model output.

## Selected local profile

The selected high-throughput profile is
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) at immutable revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. It uses BF16 weights, vLLM 0.27.1,
xgrammar 0.2.3, greedy non-thinking generation with a fixed seed, the native sampler,
a maximum batch of 16, and prefix caching. Analysis models require safetensors-only
weights; repositories containing Python files or `auto_map` hooks are rejected, and
`trust_remote_code` remains false.

| Metric | Observed | Regression gate |
|---|---:|---:|
| Valid JSON responses | `1.0` | `1.0` |
| Accepted candidate rate | `1.0` | `0.95` |
| Fully grounded response rate | `1.0` | `0.95` |
| Macro candidate F1 | `0.618132` | `0.55` |
| Claim F1 | `0.583333` | `0.5` |
| Concept F1 | `0.809524` | `0.7` |
| Relation F1 | `0.461538` | `0.35` |

On an RTX 5080 with PyTorch 2.13.0 and CUDA 13.0, all nine requests produced 3,865
tokens in one 17.23-second batch. The evaluator apportions shared batch time across
items, so its reported 1.91-second p50 is a throughput measurement, not independent
single-request latency. Startup took 19.63 seconds on the warmed machine. These are
one machine's observations and are not performance gates.

The portable Transformers BF16 path remains supported. On the same benchmark it
scored `0.581746` macro F1 and took 146.86 seconds. Dynamic FP8 under vLLM reduced
inference to 12.69 seconds but also reduced macro F1 to `0.594877` by adding false
positives, so BF16 remains the quality-first default. A 9B NF4 bakeoff also failed to
beat the 4B model; size alone is not a selection criterion.

## Reproduce the CUDA gate

Install the optional Linux/NVIDIA runtime:

```bash
uv sync --extra analysis-vllm
```

The first model fetch must be explicit. It downloads public model files only and
does not read or upload document text:

```bash
corpusdock analysis-eval benchmarks/analysis-v1/cases.json \
  --analysis-runtime vllm --allow-model-download \
  --device cuda --dtype bfloat16 --json
```

After caching, omit `--allow-model-download` and optionally pin the reported
revision. CorpusDock forces the model registry offline, disables runtime usage
telemetry, resolves the local snapshot, and performs the complete evaluation on the
workstation:

```bash
corpusdock analysis-eval benchmarks/analysis-v1/cases.json \
  --analysis-runtime vllm \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --device cuda --dtype bfloat16 --json
```

The vLLM default uses the native sampler, avoiding an nvcc or system CUDA-toolkit
dependency. Its engine, xgrammar backend, sampler, batch size, prefix-cache setting,
dtype, model revision, SaT version, and GPU memory fraction are persisted in run
provenance and therefore participate in resume compatibility.

For the portable fallback instead install `.[analysis]`, omit
`--analysis-runtime vllm`, and select an appropriate CPU or CUDA device. The optional
`.[analysis,analysis-cuda]` extra exposes Transformers bitsandbytes modes for model
bakeoffs; those options do not apply to vLLM.

## Scope and limitations

This benchmark tests extraction-contract behavior, not whether a source claim is
true, and nine synthetic passages cannot establish broad-domain quality. Accepted,
exactly grounded model output still requires cross-evidence canonicalization and
human review before it becomes a semantic knowledge graph. A model/runtime profile
must pass this public gate before a bounded private-corpus pilot. Private passages,
generated candidates, databases, reports, and model weights remain ignored local
state.
