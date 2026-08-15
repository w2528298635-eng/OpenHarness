# RepoPilot SWE-bench Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible four-arm SWE-bench Verified evaluation with
leakage-safe localization metrics, paired statistics, resumable Docker
execution, evidence reports, and a RepoPilot-first README.

**Architecture:** Keep the existing local evaluation intact and add an isolated
`openharness.repopilot.swebench` package. Public inference records and gold
evaluation records are separate models and directories; orchestration joins
them only after inference artifacts are sealed. The official SWE-bench harness
remains the authority for resolved status.

**Tech Stack:** Python 3.11, Pydantic 2, Typer, asyncio, Git, Docker Desktop/WSL
2, official SWE-bench harness, DeepSeek V4 Flash, pytest, standard-library
statistics and hashing.

## Global Constraints

- Dataset: `SWE-bench/SWE-bench_Verified`, revision recorded in every manifest.
- Sample: easy 10, medium 15, hard 20, seed `20260730`.
- Formal arms: native, legacy commit
  `15fb5947bff15fccb2faea186240fcd76ec0e2ab`, upgraded without retrieval,
  upgraded with retrieval.
- Formal repetitions: 3.
- Formal provider model: `deepseek-v4-flash`.
- Gold patch, test patch, and gold labels never enter inference models or
  prompts.
- Task is the independent statistical unit.
- Formal task selection and settings are immutable after the manifest is
  frozen.
- No release, merge, or tag before owner review.

---

### Task 1: Typed SWE-bench Contracts and Deterministic Sampling

**Files:**
- Create: `src/openharness/repopilot/swebench/__init__.py`
- Create: `src/openharness/repopilot/swebench/models.py`
- Create: `src/openharness/repopilot/swebench/sampler.py`
- Create: `tests/test_repopilot/test_swebench_models.py`
- Create: `tests/test_repopilot/test_swebench_sampler.py`

**Interfaces:**
- Consumes: plain public dataset rows as mappings.
- Produces: `PublicInstance`, `DifficultyStratum`, `SampleManifest`,
  `SamplingConfig`, `sample_manifest(rows, config)`, and
  `manifest_sha256(manifest)`.

- [ ] **Step 1: Write failing model and sampler tests**

```python
def test_public_instance_rejects_gold_fields():
    with pytest.raises(ValueError, match="gold-only"):
        PublicInstance.model_validate({"instance_id": "x", "patch": "secret"})


def test_sample_manifest_is_reproducible_and_balanced(rows):
    config = SamplingConfig(easy=2, medium=2, hard=2, seed=20260730)
    first = sample_manifest(rows, config, dataset_revision="abc")
    second = sample_manifest(reversed(rows), config, dataset_revision="abc")
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.sha256 == second.sha256
```

- [ ] **Step 2: Run tests and confirm missing-module failure**

Run:

```powershell
pytest tests/test_repopilot/test_swebench_models.py tests/test_repopilot/test_swebench_sampler.py -q
```

Expected: collection fails because `openharness.repopilot.swebench` does not
exist.

- [ ] **Step 3: Implement minimal typed contracts and sampler**

Use strict Pydantic models. Normalize official difficulty values into
`easy`, `medium`, and `hard`; sort input rows before seeded repository
round-robin sampling; canonicalize JSON with sorted keys before SHA-256.
Raise `InsufficientStratumError` with requested and available counts.

- [ ] **Step 4: Run focused tests**

Run the Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/openharness/repopilot/swebench tests/test_repopilot/test_swebench_models.py tests/test_repopilot/test_swebench_sampler.py
git commit -m "feat(repopilot): add frozen SWE-bench sampling"
```

### Task 2: Dataset Loading and Frozen Manifest Persistence

**Files:**
- Create: `src/openharness/repopilot/swebench/dataset.py`
- Create: `tests/test_repopilot/test_swebench_dataset.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: a `DatasetProvider` protocol and `SamplingConfig`.
- Produces: `prepare_manifest(provider, output_path, config) -> SampleManifest`
  and `JsonDatasetProvider` for offline fixtures.

- [ ] **Step 1: Write failing tests**

Test that the provider revision is persisted, manifest writes are atomic,
existing mismatched manifests require `force=True`, and gold fields are removed
before `PublicInstance` validation.

- [ ] **Step 2: Verify red**

```powershell
pytest tests/test_repopilot/test_swebench_dataset.py -q
```

Expected: import failure for `swebench.dataset`.

- [ ] **Step 3: Implement providers and persistence**

Keep Hugging Face loading behind a lazy optional import so core RepoPilot does
not require `datasets`. Add `.openharness/swebench/` and local dataset caches to
`.gitignore`.

- [ ] **Step 4: Verify green**

Run the Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add .gitignore src/openharness/repopilot/swebench/dataset.py tests/test_repopilot/test_swebench_dataset.py
git commit -m "feat(repopilot): persist public SWE-bench manifests"
```

### Task 3: Leakage-Safe Gold Labels and Localization Metrics

**Files:**
- Create: `src/openharness/repopilot/swebench/gold.py`
- Create: `src/openharness/repopilot/swebench/localization.py`
- Create: `tests/test_repopilot/test_swebench_gold.py`
- Create: `tests/test_repopilot/test_swebench_localization.py`

**Interfaces:**
- Consumes: post-inference gold patch, base/patched source text, and ranked
  `RetrievedLocation` records.
- Produces: `GoldLabels`, `LocalizationMetrics`,
  `extract_gold_files(patch)`, `extract_gold_symbols(...)`, and
  `score_localization(labels, ranking, ks=(1, 3, 5, 10))`.

- [ ] **Step 1: Write hand-computed failing metric tests**

```python
def test_file_metrics_use_deduplicated_best_chunk():
    labels = GoldLabels(files=["a.py", "b.py"])
    ranking = [
        RetrievedLocation(file="x.py", rank=1, characters=100),
        RetrievedLocation(file="a.py", rank=2, characters=100),
        RetrievedLocation(file="a.py", rank=3, characters=100),
        RetrievedLocation(file="b.py", rank=4, characters=100),
    ]
    result = score_localization(labels, ranking, ks=(1, 3, 5))
    assert result.recall_at[1] == 0
    assert result.recall_at[3] == 0.5
    assert result.recall_at[5] == 1
    assert result.mrr == 0.5
```

Also test rename paths, `/dev/null`, test-patch exclusion, Windows separators,
async functions, nested functions, classes, module-level changes, invalid
Python, and explicit denominators.

- [ ] **Step 2: Verify red**

```powershell
pytest tests/test_repopilot/test_swebench_gold.py tests/test_repopilot/test_swebench_localization.py -q
```

Expected: import failures.

- [ ] **Step 3: Implement patch parsing, AST attribution, and metrics**

Parse unified diff headers without applying the patch. Accept optional patched
source for additions that cannot map to the base AST. Deduplicate file ranks,
compute binary-relevance DCG, and estimate context tokens as
`ceil(characters / 4)`.

- [ ] **Step 4: Verify green**

Run the Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/openharness/repopilot/swebench/gold.py src/openharness/repopilot/swebench/localization.py tests/test_repopilot/test_swebench_gold.py tests/test_repopilot/test_swebench_localization.py
git commit -m "feat(repopilot): measure SWE-bench localization"
```

### Task 4: Paired Statistics and Honest Claims

**Files:**
- Create: `src/openharness/repopilot/swebench/statistics.py`
- Create: `tests/test_repopilot/test_swebench_statistics.py`

**Interfaces:**
- Consumes: paired per-task observations keyed by instance ID.
- Produces: `PairedComparison`, `paired_bootstrap`,
  `paired_permutation_test`, `mcnemar_exact`, `holm_adjust`, and
  `claim_classification`.

- [ ] **Step 1: Write failing deterministic statistical tests**

Test zero difference, known all-win difference, reproducible bootstrap seed,
exact McNemar values for small discordant counts, Holm monotonicity, missing
pair rejection, and the rule that significance requires both adjusted
`p < 0.05` and a confidence interval excluding zero.

- [ ] **Step 2: Verify red**

```powershell
pytest tests/test_repopilot/test_swebench_statistics.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement standard-library statistical functions**

Enumerate exact sign flips when the pair count is small; otherwise use a
seeded Monte Carlo permutation count declared in the report. Implement
two-sided exact binomial McNemar without SciPy.

- [ ] **Step 4: Verify green**

Run the Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/openharness/repopilot/swebench/statistics.py tests/test_repopilot/test_swebench_statistics.py
git commit -m "feat(repopilot): add paired evaluation statistics"
```

### Task 5: Docker Doctor and Official Harness Boundary

**Files:**
- Create: `src/openharness/repopilot/swebench/docker_runner.py`
- Create: `tests/test_repopilot/test_swebench_docker_runner.py`

**Interfaces:**
- Consumes: an injectable `CommandRunner`, predictions, cache path, and harness
  settings.
- Produces: `DoctorReport`, `HarnessPrediction`, `HarnessResult`,
  `run_doctor()`, `write_predictions_jsonl()`, and
  `OfficialHarnessRunner.evaluate(...)`.

- [ ] **Step 1: Write failing tests**

Test Docker-daemon-unavailable diagnosis, WSL warning on Windows, CPU/RAM/disk
thresholds, JSONL field names, command argv construction without shell
interpolation, timeout classification, and harness result parsing.

- [ ] **Step 2: Verify red**

```powershell
pytest tests/test_repopilot/test_swebench_docker_runner.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement doctor and harness wrapper**

Use subprocess argv arrays only. Never call prune or delete Docker resources.
Make thresholds explicit: x86_64, 8 logical CPUs recommended, 16 GB RAM
recommended, and 120 GB free Docker storage required for formal local runs.

- [ ] **Step 4: Verify green**

Run the Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/openharness/repopilot/swebench/docker_runner.py tests/test_repopilot/test_swebench_docker_runner.py
git commit -m "feat(repopilot): add SWE-bench Docker boundary"
```

### Task 6: Four-Arm Adapter and Immutable Inference Artifacts

**Files:**
- Create: `src/openharness/repopilot/swebench/adapters.py`
- Create: `src/openharness/repopilot/swebench/inference.py`
- Create: `tests/test_repopilot/test_swebench_adapters.py`
- Create: `tests/test_repopilot/test_swebench_inference.py`

**Interfaces:**
- Consumes: `PublicInstance`, `InferenceBudget`, repository path, and injected
  native/legacy/current runners.
- Produces: `EvaluationArm`, `AgentAdapter`, `InferenceRequest`,
  `InferenceArtifact`, and `InferenceRunner.run`.

- [ ] **Step 1: Write failing contract and leakage tests**

Assert every adapter returns the same artifact schema, the D/C configurations
differ only in retrieval, the legacy commit is exact, deprecated model aliases
are rejected, output patch hashes are stable, and no inference serialization
can contain any gold-only field name.

- [ ] **Step 2: Verify red**

```powershell
pytest tests/test_repopilot/test_swebench_adapters.py tests/test_repopilot/test_swebench_inference.py -q
```

Expected: import failures.

- [ ] **Step 3: Implement adapters and sealed artifacts**

Reuse OpenHarness QueryEngine for native execution and RepoPilotService for
current execution. Run legacy through a pinned Git worktree and compatibility
entry point. Write artifact JSON to a temporary file, fsync, replace, and then
write its SHA-256 seal.

- [ ] **Step 4: Verify green**

Run the Step 2 command. Expected: all pass with fake runners.

- [ ] **Step 5: Commit**

```powershell
git add src/openharness/repopilot/swebench/adapters.py src/openharness/repopilot/swebench/inference.py tests/test_repopilot/test_swebench_adapters.py tests/test_repopilot/test_swebench_inference.py
git commit -m "feat(repopilot): add fair SWE-bench inference arms"
```

### Task 7: Checkpointed Orchestration and Reports

**Files:**
- Create: `src/openharness/repopilot/swebench/orchestration.py`
- Create: `src/openharness/repopilot/swebench/reporting.py`
- Create: `tests/test_repopilot/test_swebench_orchestration.py`
- Create: `tests/test_repopilot/test_swebench_reporting.py`

**Interfaces:**
- Consumes: manifest, arms, repetitions, adapters, harness runner, evaluator.
- Produces: `RunKey`, `ExperimentCheckpoint`, `ExperimentOrchestrator`,
  `ExperimentReport`, JSON/CSV/Markdown evidence.

- [ ] **Step 1: Write failing resume and report tests**

Test idempotent run keys, skip-only-verified completion, infrastructure retries
separate from agent attempts, cancellation at run boundaries, explicit
denominators, difficulty breakdowns, primary comparisons, and no significant
claim for an interval crossing zero.

- [ ] **Step 2: Verify red**

```powershell
pytest tests/test_repopilot/test_swebench_orchestration.py tests/test_repopilot/test_swebench_reporting.py -q
```

Expected: import failures.

- [ ] **Step 3: Implement checkpoint state machine and rendering**

Use atomic JSON checkpoint writes and bounded `asyncio.Semaphore` concurrency.
Persist every status transition. Join gold labels only after confirming the
inference artifact seal.

- [ ] **Step 4: Verify green**

Run the Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/openharness/repopilot/swebench/orchestration.py src/openharness/repopilot/swebench/reporting.py tests/test_repopilot/test_swebench_orchestration.py tests/test_repopilot/test_swebench_reporting.py
git commit -m "feat(repopilot): orchestrate resumable SWE-bench runs"
```

### Task 8: CLI, Configuration, and CI

**Files:**
- Create: `src/openharness/repopilot/swebench/cli.py`
- Create: `tests/test_repopilot/test_swebench_cli.py`
- Modify: `src/openharness/repopilot/cli.py`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces the six approved `openh repopilot swebench` commands.

- [ ] **Step 1: Write failing Typer contract tests**

Test command help, `doctor` JSON output, offline manifest preparation,
pilot/formal guards, resume, report rendering, deprecated model rejection, and
refusal to start formal inference when doctor is not ready.

- [ ] **Step 2: Verify red**

```powershell
pytest tests/test_repopilot/test_swebench_cli.py -q
```

Expected: command not found.

- [ ] **Step 3: Add CLI and optional dependencies**

Add a `swebench` optional group for official harness/dataset dependencies while
keeping core imports lazy. Update CI to install both `dev` and `api` extras and
run SWE-bench unit tests without Docker.

- [ ] **Step 4: Verify green and CLI help**

```powershell
pytest tests/test_repopilot/test_swebench_cli.py -q
openh repopilot swebench --help
```

Expected: tests pass and all six commands appear.

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml .github/workflows/ci.yml src/openharness/repopilot/cli.py src/openharness/repopilot/swebench/cli.py tests/test_repopilot/test_swebench_cli.py
git commit -m "feat(repopilot): expose SWE-bench evaluation CLI"
```

### Task 9: Full Verification and Three-Instance Calibration

**Files:**
- Create: `docs/evidence/swebench/pilot-manifest.json`
- Create: `docs/evidence/swebench/pilot-report.json`
- Create: `docs/evidence/swebench/pilot-report.md`
- Modify tests or implementation only when a reproduced failure has a failing
  regression test first.

**Interfaces:**
- Produces a measured formal-run projection or a precise environment blocker.

- [ ] **Step 1: Run unit and integration verification**

```powershell
pytest tests/test_repopilot -q
ruff check src/openharness/repopilot tests/test_repopilot
```

- [ ] **Step 2: Run Docker doctor**

```powershell
openh repopilot swebench doctor --json
```

Expected: machine facts and actionable readiness, without modifying Docker.

- [ ] **Step 3: Validate official harness with one gold patch**

Use one calibration instance and `max_workers=1`. Record harness version,
Docker image identity, duration, and disk change.

- [ ] **Step 4: Run three calibration instances across four arms**

Use one excluded instance per difficulty stratum. Record model tokens, costs,
duration, Docker storage, failures, and retry behavior.

- [ ] **Step 5: Generate and commit calibration evidence**

The report must project 540-run cost and duration as a range and state whether
formal local execution is ready.

### Task 10: Formal Evidence, README, and Release Candidate

**Files:**
- Create: `docs/evidence/swebench/formal-manifest.json`
- Create: `docs/evidence/swebench/formal-report.json`
- Create: `docs/evidence/swebench/formal-report.csv`
- Create: `docs/evidence/swebench/formal-report.md`
- Create: `scripts/demo_repopilot.ps1`
- Modify: `README.md`
- Modify: `docs/repopilot-evaluation.md`
- Create: `docs/releases/repopilot-v1.0.0-rc1.md`

**Interfaces:**
- Produces the owner-reviewable release candidate; it does not publish it.

- [ ] **Step 1: Freeze the 45-instance manifest and configuration**

Persist dataset revision, manifest digest, exact commits, provider model,
budgets, concurrency, price snapshot, and harness version.

- [ ] **Step 2: Run or resume all 540 inference entries**

Do not silently replace missing or failed entries. Emit periodic checkpoint
summaries.

- [ ] **Step 3: Evaluate every sealed patch with the official harness**

Generate complete denominators and classify agent, infrastructure, and dataset
failures.

- [ ] **Step 4: Generate quantitative and statistical evidence**

Render overall and difficulty-stratified repair/localization results, primary
paired comparisons, adjusted p-values, confidence intervals, costs, timings,
and representative failures.

- [ ] **Step 5: Write RepoPilot-first README and one-click demo**

All numeric claims must be sourced from committed formal evidence. Include the
SWE-bench Verified limitations and exact reproduction commands.

- [ ] **Step 6: Run final verification**

```powershell
pytest tests/test_repopilot -q
ruff check src/openharness/repopilot tests/test_repopilot
git diff --check
git status --short
```

- [ ] **Step 7: Commit release candidate and stop for owner review**

Do not merge, tag, push a release, or claim significance beyond the report.

