# Planned dual retrieval: three-task development pilot

This pre-multi-query development record covers three frozen tasks from `SWE-bench/SWE-bench_Verified`, one
from each local difficulty stratum. It evaluates **localization**, not end-to-end
patch correctness. The three-task sample is a development diagnostic and is too
small for a broad benchmark or statistical-significance claim.

## Compared retrieval paths

- Lexical baseline: TF-IDF-style term matching with symbol/path boosts.
- Previous candidate rerank: semantic scores are computed only inside the
  lexical candidate set.
- Planned dual structural: a Query Planner produces code-oriented queries;
  lexical top-100 and independently computed dense top-100 are normalized and
  fused; same-file definitions and symbol references then expand the context.

## Observed aggregate results

| Method | Recall@5 | Hit@5 | MRR | Irrelevant context | Mean retrieval |
|---|---:|---:|---:|---:|---:|
| Lexical baseline | 33.33% | 33.33% | 0.333 | 88.87% | 4.74 s |
| Previous candidate rerank | 33.33% | 33.33% | 0.333 | 89.94% | 54.32 s |
| Planned dual structural, cached | 66.67% | 66.67% | 0.444 | 81.96% | 32.60 s |
| Planned dual structural, first build | 66.67% | 66.67% | 0.444 | 81.96% | 413.55 s |

The new path located the scikit-learn gold file at rank 1 and the matplotlib
gold file at rank 3; it missed the Django gold file in the top five. The large
first-build latency is the cost of encoding previously unseen chunks. The
shared SQLite embedding cache reduced the observed rerun retrieval mean to
32.60 seconds without changing rankings.

## Interpretation boundary

These results show that independent dense recall can recover a file absent from
the lexical candidate set in this pilot. They do **not** isolate which of query
planning, dual recall, or structure expansion contributed each gain. The code
therefore exposes ablation switches for those stages, and the 45-task run is
reported separately when complete. Because the public tasks were used during
development, that larger result is also development evidence rather than an
untouched confirmatory test.

The current implementation subsequently added content-addressed cross-revision
embedding reuse, dense multi-query aggregation, and caller-first structural
expansion. Its final 45-task checkpoint is reported separately rather than
silently replacing these earlier observations.

Machine-readable values are in
[`dual-retrieval-pilot3-summary.json`](dual-retrieval-pilot3-summary.json).
