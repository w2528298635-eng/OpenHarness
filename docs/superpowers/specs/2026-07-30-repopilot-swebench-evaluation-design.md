# RepoPilot SWE-bench Evaluation Design

## Status

Approved by the project owner on 2026-07-30. This document freezes the
evaluation intent before implementation and formal model runs.

## Goal

Build a reproducible, leakage-resistant SWE-bench evaluation that answers three
questions with quantitative evidence:

1. Does the upgraded RepoPilot outperform a native OpenHarness ReAct agent?
2. Does the upgraded RepoPilot outperform the pre-upgrade RepoPilot?
3. Does RepoPilot retrieval improve localization quality and final repair
   outcomes compared with the same upgraded system without retrieval?

The evaluation must report negative and inconclusive results honestly. A
statistically significant improvement is a result to measure, not a result the
implementation may assume.

## Dataset and Frozen Sample

The source dataset is `SWE-bench/SWE-bench_Verified`. Its human difficulty
annotations are mapped into three strata:

| RepoPilot stratum | SWE-bench Verified difficulty | Samples |
| --- | --- | ---: |
| easy | `<15 min fix` | 10 |
| medium | `15 min - 1 hour` | 15 |
| hard | `1-4 hours` and `>4 hours` | 20 |

Sampling is deterministic:

- seed: `20260730`;
- stratify by difficulty and repository;
- use seeded round-robin selection across repositories to limit repository
  dominance;
- persist dataset identity, revision, instance IDs, source difficulty, and
  repository in a public manifest;
- store a SHA-256 digest of the canonical manifest;
- never remove or replace a formal task after the manifest is frozen.

Three additional instances, one per stratum and excluded from the formal
manifest, are used for environment and budget calibration.

## Experimental Arms

Each formal instance is run three times under four arms, for 540 attempted
model runs:

| Arm | System |
| --- | --- |
| A | Native OpenHarness ReAct with no RepoPilot scheduler |
| B | Legacy RepoPilot at commit `15fb5947bff15fccb2faea186240fcd76ec0e2ab` |
| C | Upgraded RepoPilot with retrieval disabled |
| D | Upgraded RepoPilot with retrieval enabled |

The legacy arm may use an external SWE-bench compatibility adapter, but the
adapter must not change its scheduling, recovery, prompt, or tool policy.

All arms receive the same:

- repository and base commit;
- issue statement;
- provider and exact model version;
- sampling and reasoning settings;
- model-call, token, and wall-clock budgets;
- Docker resource and network policy;
- final SWE-bench harness evaluator.

The formal provider is `deepseek-v4-flash`; deprecated aliases such as
`deepseek-chat` are not accepted in a frozen run. Exact provider settings and
the release-candidate commit are stored in the run manifest.

Differences intrinsic to the scaffold remain treatment variables. In
particular, native ReAct may decide whether to run available public checks,
while RepoPilot performs deterministic public verification.

## Leakage Boundary

Inference receives only public task information and repository contents at the
base commit. It must not receive:

- the gold patch;
- `test_patch`;
- `FAIL_TO_PASS` answers;
- gold changed-file or symbol labels;
- metrics computed from gold data.

Gold data is loaded only by a separate post-inference evaluator after ranked
retrieval output and the model patch have been persisted. The evaluator records
hashes of these artifacts before reading gold data. Automated tests must prove
that inference request models cannot contain gold-only fields.

Agent-internal verification may use syntax checks, scope checks, and tests
already present at the base commit. The official harness applies its evaluation
assets only after inference and is the sole authority for `resolved`.

## SWE-bench Integration

Add a focused `openharness.repopilot.swebench` package:

- `models.py`: manifests, run keys, prediction records, localization metrics,
  harness results, and reports;
- `dataset.py`: public dataset loading and revision capture;
- `sampler.py`: deterministic repository-aware sampling and manifest hashing;
- `gold.py`: post-inference gold file and Python-symbol extraction;
- `adapters.py`: a common interface for the four inference arms;
- `docker_runner.py`: environment diagnostics and the pinned official harness;
- `inference.py`: patch generation and immutable inference artifacts;
- `localization.py`: file, symbol, ranking, and context-efficiency metrics;
- `statistics.py`: paired estimates, confidence intervals, hypothesis tests,
  and multiplicity correction;
- `reporting.py`: JSON, JSONL, CSV, and Markdown evidence;
- `orchestration.py`: checkpointed task/arm/repetition execution and resume.

The CLI exposes:

```text
openh repopilot swebench doctor
openh repopilot swebench prepare
openh repopilot swebench pilot
openh repopilot swebench run
openh repopilot swebench resume
openh repopilot swebench report
```

Each `(instance, arm, repetition)` is an idempotent run key and checkpoint.
Completed entries are never silently rerun. Infrastructure retries are recorded
separately from agent attempts.

Large repositories, Docker layers, datasets, and raw run directories remain
outside Git. The repository stores the frozen manifest, configuration, summary
evidence, selected sanitized traces, and commands needed to reproduce results.

## Localization Ground Truth

Gold file labels are paths modified by the gold patch after excluding test-only
evaluation assets. File rankings are deduplicated by the best-ranked chunk for
each path.

For parseable Python files, changed hunk lines are mapped to the smallest
enclosing function, async function, or class in the base or patched AST.
Unparseable files and module-level changes remain eligible for file metrics but
are excluded from symbol-metric denominators. Reports must show every
denominator.

Primary localization metrics:

- file Recall@1, @3, @5, and @10;
- file Hit@1, @3, @5, and @10;
- Precision@K;
- mean reciprocal rank;
- nDCG@K;
- symbol Recall@K for eligible tasks;
- first relevant file and symbol rank;
- retrieval latency;
- injected characters and estimated tokens;
- irrelevant-context rate;
- relevant-file hits per 1,000 context tokens;
- tool calls and file reads before the first gold-file read.

RepoPilot retrieval is also compared offline with the public SWE-bench BM25
retrieval data at a matched context budget. That comparison does not call the
model and does not use oracle retrieval as a candidate system.

## Repair and Efficiency Metrics

The official harness supplies:

- attempted, completed, patch-applied, and resolved counts;
- resolution rate;
- per-instance harness status and logs.

RepoPilot telemetry adds:

- scope compliance;
- no-diff rate;
- first-attempt repair rate;
- repair and replan counts;
- recovery success after an initial failure;
- model calls;
- input, output, cache-hit, and total tokens;
- estimated cost using a versioned price snapshot;
- wall-clock duration;
- infrastructure failure and retry counts.

Results are reported overall and by difficulty stratum. Per-repository results
are diagnostic only because the sample is not powered for repository-level
claims.

## Statistical Analysis

The task is the independent statistical unit. Three repetitions are summarized
into a per-task success proportion before cross-system inference.

Pre-registered primary comparisons:

1. C versus A for scheduler/platform value;
2. C versus B for upgrade value;
3. D versus C for retrieval value.

Report:

- paired absolute and relative differences;
- task-level paired bootstrap 95% confidence intervals;
- a paired permutation test for per-task success proportions;
- McNemar's exact test for a declared binary task outcome where applicable;
- Holm-adjusted p-values for the three primary hypotheses;
- effect sizes and discordant-task tables.

The phrase "statistically significant" is permitted only when the Holm-adjusted
`p < 0.05` and the corresponding 95% confidence interval excludes zero.
Otherwise the result is described as directional, null, or inconclusive.

No formal-task result may be used to tune prompts, retrieval weights, budgets,
or task selection. Changes after a formal run require a new versioned
experiment and must preserve the prior result.

## Failure Semantics

Failures are separated into:

- agent failures: timeout, no patch, invalid patch, verification failure, or
  official unresolved result;
- infrastructure failures: Docker/network/provider interruption or disk
  exhaustion;
- dataset/harness failures: image build or official task defects.

Agent failures count in the relevant outcome. Infrastructure failures may be
retried under a declared policy and remain visible. Missing results are never
converted into failures or removed silently. Reports list exclusions, retries,
and denominators.

## Resource and Execution Gates

Before downloading the full sample:

1. `doctor` verifies Docker Engine, WSL 2 on Windows, architecture, CPU, RAM,
   writable cache, and at least 120 GB available Docker storage;
2. the official gold-patch smoke test validates one container;
3. three non-formal calibration instances measure tokens, duration, disk, and
   harness reliability;
4. calibration produces a projected formal-run cost and duration.

Formal inference starts only with a functioning environment and sufficient API
balance. It may run unattended with bounded concurrency, checkpointing, resume,
and periodic progress summaries.

## Tests

Use test-driven development for:

- deterministic sampling and manifest hashing;
- repository balancing and insufficient-stratum errors;
- unified adapter contracts;
- gold patch parsing and path normalization;
- Python symbol attribution and denominator handling;
- every localization metric with hand-computed fixtures;
- gold-data leakage prevention;
- run-key idempotency and resume;
- infrastructure-versus-agent failure classification;
- paired statistics, bootstrap determinism, and Holm correction;
- official prediction JSONL serialization;
- CLI contracts and report rendering.

Integration tests use fake datasets, fake providers, and a fake harness. Docker
tests are separately marked and run only after `doctor` succeeds.

## README and Release

After evidence exists, the root README becomes RepoPilot-first and includes:

- a one-sentence value proposition and 30-second quick start;
- architecture and run-flow diagrams;
- a four-arm fairness matrix;
- the frozen dataset manifest and digest;
- resolved, localization, cost, latency, and safety metrics with confidence
  intervals;
- a complete trace and representative failures;
- reproduction commands and a one-click local demo;
- explicit SWE-bench Verified limitations;
- OpenHarness attribution and a precise list of RepoPilot additions.

The README must not contain placeholder wins or claims unsupported by committed
evidence. The owner reviews the release candidate, metrics, limitations,
release notes, and version number before merge, tag, or formal publication.

