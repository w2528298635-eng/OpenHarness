# Planned independent dual retrieval: 45-task result

This development evaluation covers the frozen 45-task manifest from
`SWE-bench/SWE-bench_Verified`: 10 locally stratified easy tasks, 15 medium,
and 20 hard. Gold patches were withheld from queries and retrieval, then used
post hoc to score whether the retrieved context contained the files modified by
the reference patch.

## Overall localization quality

| Method | Recall@1 | Recall@5 | Hit@5 | MRR | Irrelevant context | Mean retrieval |
|---|---:|---:|---:|---:|---:|---:|
| Lexical baseline | 11.11% | 15.56% | 15.56% | 0.133 | 89.69% | 3.41 s |
| Semantic rerank inside lexical candidates | 13.33% | 21.11% | 22.22% | 0.178 | 85.84% | 49.09 s |
| **Planned independent lexical + dense** | **15.56%** | **28.89%** | **33.33%** | **0.239** | **85.12%** | 27.72 s |
| Planned dual + structural expansion | 15.56% | 24.44% | 26.67% | 0.193 | 87.99% | 29.18 s |

Against lexical retrieval, the recommended path improved Recall@5 by 13.33
percentage points (85.7% relative), Hit@5 by 17.78 points (114.3% relative),
and MRR by 79.2% relative. Irrelevant context fell by 4.57 percentage points.
Against the older candidate-set reranker, Recall@5 rose 36.8%, Hit@5 50.0%,
and MRR 34.4% relative. The quality gain comes with more latency than lexical
retrieval; cached dual retrieval averaged 27.72 seconds per task on this machine.

## Recommended path by local difficulty stratum

| Stratum | Tasks | Recall@5 | Hit@5 | MRR | Irrelevant context |
|---|---:|---:|---:|---:|---:|
| Easy | 10 | 50.00% | 50.00% | 0.400 | 85.85% |
| Medium | 15 | 36.67% | 40.00% | 0.283 | 80.36% |
| Hard | 20 | 12.50% | 20.00% | 0.125 | 88.32% |

The hard stratum remains the largest improvement opportunity. Structural
expansion was implemented and evaluated, but it reduced the aggregate result,
so it is now opt-in rather than part of the recommended default.

## Interpretation boundary

This is a file-localization evaluation, not an official SWE-bench resolved-rate
claim and not proof that end-to-end patches improve by the same percentage.
The 45 public tasks were used while developing the retriever, so the result is
development evidence rather than an untouched confirmatory test. The frozen
manifest hash is recorded in the machine-readable
[`planned-dual-45-summary.json`](planned-dual-45-summary.json), and the
[three-task component ablation](retrieval-ablation-pilot3.md) separately shows
which stages produced the observed pilot gains.
