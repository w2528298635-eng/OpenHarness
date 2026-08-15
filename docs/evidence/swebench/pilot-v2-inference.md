# RepoPilot SWE-bench pilot-v2 inference record

Observed on 2026-07-30 with the frozen three-task calibration manifest
(`238a62a8ba8e9ea79734c3a98b877d11606ecb7c8892f225549d42f6d223ee91`).
This is an inference record, not an official SWE-bench resolution report.

## Contract

- Provider/model: DeepSeek `deepseek-v4-flash` through the OpenAI-compatible
  endpoint.
- Matrix: native OpenHarness, frozen legacy RepoPilot, upgraded RepoPilot
  without retrieval, upgraded RepoPilot with retrieval; one repetition each.
- Budget per run: at most 8 model calls, 200,000 tokens, and 900 seconds.
- Input manifest excludes all gold patches, evaluator tests, and derived gold
  labels.
- Every model output was saved as a sealed inference artifact before any
  evaluator decision.

## Inference result

| Outcome | Runs | Meaning |
| --- | ---: | --- |
| `completed` | 2 | Candidate patch produced and then resolved by the official evaluator. |
| `agent_failed` | 10 | The agent did not produce an evaluator-ready result under the fixed budget. |
| Officially resolved | 2 | Both are the same scikit-learn task under the two upgraded arms. |

The two officially evaluated artifacts are both for
`scikit-learn__scikit-learn-13439`:

| Arm | Model calls | Input tokens | Output tokens | Inference duration | Official result |
| --- | ---: | ---: | ---: | ---: | --- |
| Upgraded, no retrieval | 3 | 98,262 | 2,220 | 44.51 s | Resolved 1/1; adds `Pipeline.__len__` in `sklearn/pipeline.py`. |
| Upgraded, retrieval | 3 | 160,740 | 2,242 | 64.58 s | Resolved 1/1; adds `Pipeline.__len__` in `sklearn/pipeline.py`. |

The retrieval arm retrieved `sklearn/pipeline.py` at rank 1, but on this single
inference-only task it cost 62,478 more input tokens and 20.06 more seconds.
The official reports are committed as
[no-RAG JSON](pilot-v2-official-no-rag.json) and
[RAG JSON](pilot-v2-official-rag.json). They each report one submitted,
completed, and resolved instance, with zero errors. This establishes that the
retrieval arm did not regress functional correctness on this one task; it does
not establish a RAG quality improvement. A pre-registered localization metric
across the frozen 45-task subset is still required for that claim.

## Failure taxonomy

- All three legacy-arm runs failed before a model call because Windows path
  length limits prevented legacy worktree creation.
- All three native-arm runs reached the eight-turn cap. The scikit-learn run
  did create a candidate diff, but it was not terminally verified.
- The upgraded no-retrieval arm created one evaluator-pending scikit-learn
  patch; the matplotlib and Django runs stopped in the analysis phase.
- The upgraded retrieval arm created one evaluator-pending scikit-learn patch;
  matplotlib and Django exceeded the token budget during analysis.

These errors are retained as artifacts and are engineering work items, not
excluded observations. The earlier `pilot` directory is also retained as an
invalid-adapter calibration attempt; `pilot-v2` uses the repaired precheck and
inference/evaluation separation.

## Official evaluation environment

The official evaluator ran in an Ubuntu 22.04 WSL2 distribution stored on E:,
using `swebench==4.1.0` and Docker Desktop's WSL integration. Both evaluations
used one worker and the official `SWE-bench/SWE-bench_Verified` test split.
The first environment build took 8:04; the cached RAG comparison took 7:41.

No repair rate beyond this explicit 1-task-per-arm calibration, RAG uplift,
p-value, broad benchmark comparison, or performance claim is made by this
record.
