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
| `evaluation_pending` | 2 | A non-empty candidate patch was produced; official evaluation has not run. |
| `agent_failed` | 10 | The agent did not produce an evaluator-ready result under the fixed budget. |
| Officially resolved | 0 | Not measured yet; this is **not** a zero-resolution claim. |

The two evaluator-pending artifacts are both for
`scikit-learn__scikit-learn-13439`:

| Arm | Model calls | Input tokens | Output tokens | Duration | Candidate patch |
| --- | ---: | ---: | ---: | ---: | --- |
| Upgraded, no retrieval | 3 | 98,262 | 2,220 | 44.51 s | Adds `Pipeline.__len__` in `sklearn/pipeline.py`. |
| Upgraded, retrieval | 3 | 160,740 | 2,242 | 64.58 s | Adds `Pipeline.__len__` in `sklearn/pipeline.py`. |

The retrieval arm retrieved `sklearn/pipeline.py` at rank 1, but on this single
inference-only task it cost 62,478 more input tokens and 20.06 more seconds.
It must not be described as a quality improvement until the official evaluator
is available and a pre-registered localization metric is run across the frozen
45-task subset.

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

## Official-evaluation boundary

Docker Desktop, WSL2, and the Docker daemon are available on the host.
The installed Windows Python cannot run the SWE-bench harness because that
harness imports the Unix-only `resource` module, and the only currently
installed WSL distribution is Docker's internal distribution. A Linux user
distribution plus the official SWE-bench evaluator are required before the two
candidate patches can be classified as resolved or unresolved.

No repair rate, RAG uplift, p-value, benchmark comparison, or performance
claim is made by this record.
