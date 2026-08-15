# Retrieval-component ablation: three-task development pilot

This diagnostic uses the frozen three-task pilot manifest: one easy, one medium,
and one hard task from `SWE-bench/SWE-bench_Verified`. Gold patches were used
only after retrieval to score file localization. Every arm used a 12,000-character
context budget and `top_k=12`.

| Retrieval path | Planner | Structure | Recall@5 | Hit@5 | MRR | Irrelevant context |
|---|:---:|:---:|---:|---:|---:|---:|
| Lexical | off | off | 33.33% | 33.33% | 0.333 | 88.87% |
| Lexical | on | off | 33.33% | 33.33% | 0.333 | 88.87% |
| Independent lexical + dense | off | off | 66.67% | 66.67% | 0.444 | 81.96% |
| Independent lexical + dense | on | off | 66.67% | 66.67% | **0.500** | **81.48%** |
| Independent lexical + dense | on | on | 66.67% | 66.67% | 0.444 | 82.97% |

The dominant observed gain came from independent dense recall: Recall@5 rose
from 33.33% to 66.67%. Query planning did not help the lexical-only arm, but in
the dual-retrieval arm it moved a relevant result earlier (MRR 0.444 to 0.500)
and slightly reduced irrelevant context. Structural expansion did not improve
Recall@5 and reduced MRR, so it remains implemented but is disabled by default.

These are component-level observations on `n=3`, not estimates of production
repair success. Retrieval latency is also omitted from the comparison table
because cache warmth and repository indexing differed between development runs.
The larger 45-task localization result is reported separately. Machine-readable
values, including the observed timings, are in
[`retrieval-ablation-pilot3-summary.json`](retrieval-ablation-pilot3-summary.json).
