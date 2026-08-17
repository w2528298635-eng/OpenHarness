# RepoPilot Platform Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn RepoPilot into a reusable, observable, recoverable, evaluated code-repair Agent application with local retrieval, an API adapter, a second workflow, and reproducible portfolio evidence.

**Architecture:** Keep OpenHarness as the provider, ReAct, streaming, and tool-execution layer. Extract RepoPilot's deterministic domain policy into a generic workflow runtime with typed phase handlers, composite verification, recovery decisions, event persistence, and context construction. Expose the same application service through CLI and FastAPI, and prove reuse through a read-only repository-insight workflow.

**Tech Stack:** Python 3.11, Pydantic 2, asyncio, Typer, FastAPI/Starlette, pytest, Git worktrees, OpenHarness QueryEngine and ToolRegistry, OpenAI-compatible DeepSeek API.

## Global Constraints

- Support Python `>=3.10`; run development and type checks on Python 3.11.
- Do not reimplement OpenHarness QueryEngine, ToolRegistry, provider clients, or MCP.
- Do not install or execute Docker.
- Execute only trusted local Python repositories; a worktree is not a security sandbox.
- Do not automatically install dependencies, commit repair output, merge, or push.
- Preserve MIT licensing and upstream attribution.
- Never persist API keys; redact common secret patterns and bound event payload sizes.
- Never fabricate benchmark results, success rates, speedups, or cost savings.
- Keep the existing `openh repopilot run/show/resume/report/benchmark` contract compatible.
- Preserve the user's successful run worktree until an explicit retention command removes it.

---

## File Structure

New focused modules:

- `src/openharness/repopilot/events.py`: typed lifecycle events, redaction, payload bounds.
- `src/openharness/repopilot/workflow.py`: workflow definition, handler protocol, runtime loop.
- `src/openharness/repopilot/failures.py`: failure categories and recovery policy.
- `src/openharness/repopilot/verification.py`: verification check protocol and composite verifier.
- `src/openharness/repopilot/usage.py`: token detail and versioned cost estimation.
- `src/openharness/repopilot/retrieval.py`: repository indexing and lexical ranking.
- `src/openharness/repopilot/context.py`: budgeted context assembly with selection trace.
- `src/openharness/repopilot/prompt_registry.py`: named/versioned prompt templates.
- `src/openharness/repopilot/service.py`: shared application service for CLI and API.
- `src/openharness/repopilot/api.py`: FastAPI adapter.
- `src/openharness/repopilot/insight.py`: read-only repository-insight workflow.
- `src/openharness/repopilot/evaluation.py`: strategy runner and aggregate metrics.

Existing modules remain compatibility surfaces and are narrowed:

- `models.py`: task/repair models plus imports or aliases for stable public types.
- `scheduler.py`: code-repair workflow assembly and compatibility facade.
- `verifier.py`: backwards-compatible `PythonPytestVerifier` facade.
- `policy.py`: transition/budget policy compatible with recovery decisions.
- `store.py`: versioned state/events, resilient event writes, artifacts.
- `workspace.py`: compact Windows-safe paths and explicit cleanup.
- `phase_runner.py`: model-backed phase executor and detailed usage propagation.
- `prompts.py`: delegates to versioned prompt registry.
- `benchmark.py`: compatible manifest plus evaluation strategy fields.
- `cli.py`: delegates to `RepoPilotService`; adds evaluation/API/cleanup commands.

---

### Task 1: Typed Events, Summaries, and Usage Accounting

**Files:**
- Create: `src/openharness/repopilot/events.py`
- Create: `src/openharness/repopilot/usage.py`
- Modify: `src/openharness/repopilot/models.py`
- Modify: `src/openharness/repopilot/store.py`
- Test: `tests/test_repopilot/test_events.py`
- Test: `tests/test_repopilot/test_usage.py`
- Modify: `tests/test_repopilot/test_models.py`
- Modify: `tests/test_repopilot/test_store.py`

**Interfaces:**
- Produces: `RunEvent`, `RunEventKind`, `RunSummary`, `TokenUsage`,
  `ProviderPrice`, `CostEstimate`, `redact_and_bound(value, max_chars=4000)`,
  `estimate_cost(usage, price)`.
- `RunStore.append_event(event: RunEvent | BaseModel | dict[str, Any]) -> bool`
  returns false on I/O failure instead of crashing the run.

- [ ] **Step 1: Write failing event and redaction tests**

```python
def test_run_event_redacts_secrets_and_bounds_payload() -> None:
    event = RunEvent.create(
        run_id="r1",
        kind=RunEventKind.OBSERVATION,
        phase="ANALYZE",
        data={"text": "api_key=sk-secret-value " + ("x" * 5000)},
    )
    dumped = event.model_dump_json()
    assert "sk-secret-value" not in dumped
    assert "[REDACTED]" in dumped
    assert len(event.data["text"]) <= 4015
```

- [ ] **Step 2: Run the event test and verify import failure**

Run: `python -m pytest -q tests/test_repopilot/test_events.py`

Expected: FAIL because `openharness.repopilot.events` does not exist.

- [ ] **Step 3: Implement typed events and bounded redaction**

Implement enum values for run/phase/model/tool/observation/verification/transition/
checkpoint/recovery/cancellation/completion. `RunEvent.create` must stamp UTC time,
schema version `1`, and sanitize nested strings without mutating the caller's object.

- [ ] **Step 4: Write failing token and cost tests**

```python
def test_estimate_cost_separates_cache_input_and_output() -> None:
    usage = TokenUsage(input_tokens=800_000, output_tokens=200_000, cache_hit_tokens=300_000)
    price = ProviderPrice(
        provider="deepseek",
        model="deepseek-v4-flash",
        currency="CNY",
        input_per_million=1.0,
        cache_hit_input_per_million=0.02,
        output_per_million=2.0,
        version="2026-07-30",
    )
    result = estimate_cost(usage, price)
    assert result.amount == pytest.approx(0.906)
    assert result.is_estimate is True
```

- [ ] **Step 5: Implement usage and summary models**

`TokenUsage.total_tokens` must be derived when absent. `RunSummary` must include run
identity, phase, terminal reason, timestamps/duration, model/phase/tool/check counts,
repair/replan counts, usage, optional cost estimate, changed files, and artifacts.

- [ ] **Step 6: Make RunStore event writes resilient and version-aware**

Write events as one JSON object per line. Return `False` and record an in-memory
warning when append fails; state checkpoint failure remains fatal. Add a loader that
accepts existing untyped JSONL and converts known fields into `RunEvent`.

- [ ] **Step 7: Run focused tests**

Run: `python -m pytest -q tests/test_repopilot/test_events.py tests/test_repopilot/test_usage.py tests/test_repopilot/test_models.py tests/test_repopilot/test_store.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add src/openharness/repopilot/events.py src/openharness/repopilot/usage.py src/openharness/repopilot/models.py src/openharness/repopilot/store.py tests/test_repopilot
git commit -m "feat(repopilot): add typed run telemetry"
```

### Task 2: Generic Workflow Runtime

**Files:**
- Create: `src/openharness/repopilot/workflow.py`
- Test: `tests/test_repopilot/test_workflow.py`

**Interfaces:**
- Consumes: `RunEvent`, `RunEventKind`, existing `Phase`, `RepoRunState`.
- Produces:

```python
class PhaseHandler(Protocol):
    async def handle(self, context: RunContext) -> PhaseResult: ...

@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    version: str
    initial_phase: Phase
    terminal_phases: frozenset[Phase]
    handlers: Mapping[Phase, PhaseHandler]

class WorkflowRuntime:
    async def start(self, task: RepoTaskSpec) -> RepoRunState: ...
    async def resume(self, state: RepoRunState) -> RepoRunState: ...
```

- [ ] **Step 1: Write a failing deterministic workflow test**

Use three fake phases where handlers append their name to a list. Assert ordered
execution, phase-started/finished/checkpoint events, atomic save after every
transition, and stop at COMPLETE.

- [ ] **Step 2: Run the test and verify module import failure**

Run: `python -m pytest -q tests/test_repopilot/test_workflow.py`

Expected: FAIL because `workflow.py` does not exist.

- [ ] **Step 3: Implement immutable definition and validated registration**

Reject missing handlers for reachable nonterminal phases, terminal handlers, duplicate
registrations, and an initial terminal phase.

- [ ] **Step 4: Implement the runtime loop**

The runtime owns phase dispatch, lifecycle events, checkpointing, cancellation checks,
budget checks, and terminal stop. It receives callables for state creation,
transition selection, and checkpoint persistence so it has no code-repair knowledge.

- [ ] **Step 5: Add resume and cancellation tests**

Assert resume starts at the persisted phase without replaying completed handlers.
Assert a cancellation flag produces FAILED/CANCELLED after the current safe boundary.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest -q tests/test_repopilot/test_workflow.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/openharness/repopilot/workflow.py tests/test_repopilot/test_workflow.py
git commit -m "feat(repopilot): add reusable workflow runtime"
```

### Task 3: Composite Verification and Failure Policy

**Files:**
- Create: `src/openharness/repopilot/verification.py`
- Create: `src/openharness/repopilot/failures.py`
- Modify: `src/openharness/repopilot/verifier.py`
- Modify: `src/openharness/repopilot/policy.py`
- Modify: `src/openharness/repopilot/models.py`
- Test: `tests/test_repopilot/test_verification.py`
- Test: `tests/test_repopilot/test_failures.py`
- Modify: `tests/test_repopilot/test_verifier.py`
- Modify: `tests/test_repopilot/test_policy.py`

**Interfaces:**
- Produces: `FailureCategory`, `FailureRecord`, `RecoveryAction`,
  `RecoveryDecision`, `VerificationCheckResult`, `VerificationSuiteResult`,
  `VerificationCheck`, `CompositeVerifier`.
- Keeps: `PythonPytestVerifier.verify(...) -> VerificationResult`.

- [ ] **Step 1: Write failing failure-classification tests**

Parametrize syntax error, collection error, assertion failure, import/dependency
failure, timeout, permission, and provider output. Assert stable enum categories and
normalized signatures independent of paths, durations, lines, and addresses.

- [ ] **Step 2: Implement failure models and classifier**

Separate textual classification from recovery. Include source phase, retryable flag,
signature, human summary, and artifact reference.

- [ ] **Step 3: Write failing recovery-policy tests**

```python
@pytest.mark.parametrize(
    ("phase", "category", "repeated", "expected"),
    [
        (Phase.ANALYZE, FailureCategory.STRUCTURED_OUTPUT, False, RecoveryAction.RETRY),
        (Phase.VERIFY, FailureCategory.ASSERTION, False, RecoveryAction.REPAIR),
        (Phase.VERIFY, FailureCategory.ASSERTION, True, RecoveryAction.REPLAN),
        (Phase.EXECUTE, FailureCategory.OUT_OF_SCOPE_DIFF, False, RecoveryAction.STOP),
    ],
)
def test_recovery_matrix(phase, category, repeated, expected): ...
```

- [ ] **Step 4: Implement bounded FailurePolicy**

It must return data, not mutate state. Budget exhaustion always overrides retry or
recovery with STOP and a stable terminal reason.

- [ ] **Step 5: Write failing composite-verifier tests**

Create fake checks with required/optional severity. Assert ordering, skip behavior
after a fatal prerequisite, aggregate verdict, duration, bounded summaries, and full
log artifact references.

- [ ] **Step 6: Implement verification checks**

Implement diff scope, changed-Python compile, target pytest, optional regression
pytest, and diff sanity. Use argument arrays, `shell=False`, explicit timeouts, and
worktree cwd.

- [ ] **Step 7: Preserve verifier compatibility**

Delegate `PythonPytestVerifier` to a one-check suite and translate the result back to
the existing `VerificationResult`.

- [ ] **Step 8: Run focused tests**

Run: `python -m pytest -q tests/test_repopilot/test_verification.py tests/test_repopilot/test_failures.py tests/test_repopilot/test_verifier.py tests/test_repopilot/test_policy.py`

Expected: PASS.

- [ ] **Step 9: Commit**

```powershell
git add src/openharness/repopilot/verification.py src/openharness/repopilot/failures.py src/openharness/repopilot/verifier.py src/openharness/repopilot/policy.py src/openharness/repopilot/models.py tests/test_repopilot
git commit -m "feat(repopilot): add composite verification and recovery"
```

### Task 4: Migrate Code Repair to the Workflow Runtime

**Files:**
- Create: `src/openharness/repopilot/handlers.py`
- Modify: `src/openharness/repopilot/scheduler.py`
- Modify: `src/openharness/repopilot/phase_runner.py`
- Modify: `src/openharness/repopilot/report.py`
- Modify: `tests/test_repopilot/test_scheduler.py`
- Modify: `tests/test_repopilot/test_phase_runner.py`
- Modify: `tests/test_repopilot/test_report.py`
- Create: `tests/test_repopilot/test_handlers.py`

**Interfaces:**
- Consumes: `WorkflowRuntime`, `CompositeVerifier`, `FailurePolicy`.
- Produces one handler class for PRECHECK, ANALYZE, PLAN, EXECUTE, VERIFY, REPAIR,
  REPLAN; keeps `RepoPilotScheduler.start/resume` public methods unchanged.

- [ ] **Step 1: Add handler contract tests with fake phase executor**

Assert read-only phases cannot receive write tools, PLAN receives no tools, write
phases receive constrained edit tools, structured results are validated before state
mutation, and VERIFY never asks the model to decide success.

- [ ] **Step 2: Implement phase handlers**

Move phase-specific logic out of the scheduler without changing prompt contents or
tool boundaries. Handler results contain proposed state changes; checkpointing stays
in the runtime.

- [ ] **Step 3: Add usage-detail propagation tests**

Assert `OpenHarnessPhaseRunner` aggregates input/output/cache tokens from stream
usage events while retaining total-token compatibility when only a total is known.

- [ ] **Step 4: Replace scheduler loop with workflow assembly**

Keep constructor compatibility. `start` creates the workspace and run store, then
delegates to runtime. `resume` loads state and delegates without recreating worktree.

- [ ] **Step 5: Generate RunSummary at every terminal state**

Reports must consume the summary and link state, event, diff, verification, and log
artifacts. Existing report fields remain visible.

- [ ] **Step 6: Run migration regression tests**

Run: `python -m pytest -q tests/test_repopilot/test_handlers.py tests/test_repopilot/test_scheduler.py tests/test_repopilot/test_phase_runner.py tests/test_repopilot/test_report.py tests/test_repopilot/test_cli.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/openharness/repopilot/handlers.py src/openharness/repopilot/scheduler.py src/openharness/repopilot/phase_runner.py src/openharness/repopilot/report.py tests/test_repopilot
git commit -m "refactor(repopilot): run repair through workflow handlers"
```

### Task 5: Windows-Safe Worktree Lifecycle

**Files:**
- Modify: `src/openharness/repopilot/workspace.py`
- Modify: `src/openharness/repopilot/models.py`
- Modify: `src/openharness/repopilot/cli.py`
- Modify: `.gitignore`
- Modify: `tests/test_repopilot/test_workspace.py`
- Modify: `tests/test_repopilot/test_real_workspace_integration.py`
- Modify: `tests/test_repopilot/test_examples.py`

**Interfaces:**
- Produces:

```python
async def WorkspaceManager.create(repo_path: Path, run_id: str, *, base_path: Path | None = None) -> WorkspaceLease: ...
async def WorkspaceManager.cleanup(lease: WorkspaceLease, *, force: bool = False) -> None: ...
async def WorkspaceManager.prune(repo_path: Path) -> None: ...
```

- [ ] **Step 1: Reproduce both existing failures as focused tests**

Test that example copying excludes generated run/worktree artifacts. Test worktree
creation under a deliberately long repository path while the chosen worktree base
remains compact.

- [ ] **Step 2: Run and capture the expected Windows failures**

Run: `python -m pytest -q tests/test_repopilot/test_examples.py::test_discount_example_is_reproducible_and_manifests_load tests/test_repopilot/test_real_workspace_integration.py::test_real_worktree_edit_and_pytest_leave_original_untouched`

Expected before fix: one copy/path-length failure and one Git worktree failure.

- [ ] **Step 3: Implement compact workspace leases**

Default to a short per-user temp root or configured `OPENHARNESS_REPOPILOT_WORKTREE_ROOT`,
use an abbreviated run hash, persist the resolved lease in state, and validate that
cleanup targets match Git's registered worktree.

- [ ] **Step 4: Implement explicit cleanup and prune CLI**

Add `openh repopilot cleanup <run-id> --repo <repo>` and `--force`. Successful runs
are not silently deleted. Failed worktrees are retained by default.

- [ ] **Step 5: Exclude generated artifacts from examples and repository status**

Ignore `.openharness-repopilot-worktrees/`, `.openharness/repopilot/runs/`, pytest
caches, and task-local generated files without ignoring source fixture files.

- [ ] **Step 6: Run focused and all RepoPilot tests with a workspace basetemp**

Run: `python -m pytest -q tests/test_repopilot --basetemp .tmp/pytest-repopilot`

Expected: PASS with no path-length or permission errors.

- [ ] **Step 7: Commit**

```powershell
git add src/openharness/repopilot/workspace.py src/openharness/repopilot/models.py src/openharness/repopilot/cli.py .gitignore tests/test_repopilot
git commit -m "fix(repopilot): manage Windows-safe worktree lifecycle"
```

### Task 6: Versioned Prompts and Repository Retrieval

**Files:**
- Create: `src/openharness/repopilot/prompt_registry.py`
- Create: `src/openharness/repopilot/retrieval.py`
- Create: `src/openharness/repopilot/context.py`
- Modify: `src/openharness/repopilot/prompts.py`
- Modify: `src/openharness/repopilot/models.py`
- Test: `tests/test_repopilot/test_prompt_registry.py`
- Test: `tests/test_repopilot/test_retrieval.py`
- Test: `tests/test_repopilot/test_context.py`
- Modify: `tests/test_repopilot/test_prompts.py`

**Interfaces:**
- Produces `PromptTemplate`, `PromptRegistry`, `CodeChunk`, `RetrievalQuery`,
  `RetrievalResult`, `RepositoryIndex`, `ContextSelection`, `ContextBuilder`.

- [ ] **Step 1: Write failing prompt-registry tests**

Assert unknown templates/variables fail, exact versions are recorded, and rendering is
deterministic. Register ANALYZE/PLAN/EXECUTE/REPAIR/REPLAN version `2`.

- [ ] **Step 2: Implement prompt registry and compatibility wrappers**

Existing functions in `prompts.py` call the registry so current callers keep working.

- [ ] **Step 3: Write failing repository-index tests**

Use a fixture with Python functions, imports, docs, binary files, ignored directories,
and an oversized file. Assert AST symbol chunks, text fallback, ignore rules, stable
chunk IDs, and maximum sizes.

- [ ] **Step 4: Implement local index and lexical ranking**

Tokenize identifiers and natural language, score term frequency/inverse document
frequency plus exact symbol/path bonuses, and return score reasons. Do not add an
embedding dependency.

- [ ] **Step 5: Write failing context-budget tests**

Assert failure output and explicitly suspected files are prioritized, duplicate
chunks are removed, selected characters stay within budget, and every selection has
a source path, line range, score, and reason.

- [ ] **Step 6: Implement ContextBuilder and serialization**

Build prompt-ready context and a trace artifact. Add task configuration to enable
retrieval and set file/chunk/context budgets.

- [ ] **Step 7: Run focused tests**

Run: `python -m pytest -q tests/test_repopilot/test_prompt_registry.py tests/test_repopilot/test_retrieval.py tests/test_repopilot/test_context.py tests/test_repopilot/test_prompts.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add src/openharness/repopilot/prompt_registry.py src/openharness/repopilot/retrieval.py src/openharness/repopilot/context.py src/openharness/repopilot/prompts.py src/openharness/repopilot/models.py tests/test_repopilot
git commit -m "feat(repopilot): add versioned retrieval context"
```

### Task 7: Integrate Context and Deterministic Phase Execution

**Files:**
- Modify: `src/openharness/repopilot/handlers.py`
- Modify: `src/openharness/repopilot/phase_runner.py`
- Modify: `src/openharness/repopilot/store.py`
- Create: `src/openharness/repopilot/executors.py`
- Test: `tests/test_repopilot/test_executors.py`
- Modify: `tests/test_repopilot/test_handlers.py`
- Modify: `tests/test_repopilot/test_phase_runner.py`

**Interfaces:**
- Produces `PhaseExecutor` protocol, `OpenHarnessPhaseExecutor`, and
  `ScriptedPhaseExecutor`.

- [ ] **Step 1: Write scripted-executor tests**

Assert queued results are consumed by phase, unexpected calls fail loudly, call
records include prompt version/context IDs/tool names, and no provider is contacted.

- [ ] **Step 2: Implement executor seam around the existing phase runner**

Do not duplicate provider configuration. The OpenHarness executor delegates to
`OpenHarnessPhaseRunner`; the scripted executor returns validated `PhaseRunResult`.

- [ ] **Step 3: Add retrieval integration tests**

Run the same ANALYZE request with retrieval off/on. Assert off has no context trace;
on writes `context-analyze-<attempt>.json` and embeds only selected evidence.

- [ ] **Step 4: Integrate ContextBuilder into model-backed handlers**

Use retrieval for ANALYZE and REPLAN first. PLAN and write phases consume validated
prior results and may add exact target snippets without rebuilding the whole index.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest -q tests/test_repopilot/test_executors.py tests/test_repopilot/test_handlers.py tests/test_repopilot/test_phase_runner.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/openharness/repopilot/executors.py src/openharness/repopilot/handlers.py src/openharness/repopilot/phase_runner.py src/openharness/repopilot/store.py tests/test_repopilot
git commit -m "feat(repopilot): integrate traceable repair context"
```

### Task 8: Ten-Task Evaluation Suite and Strategy Runner

**Files:**
- Create: `src/openharness/repopilot/evaluation.py`
- Modify: `src/openharness/repopilot/benchmark.py`
- Modify: `src/openharness/repopilot/cli.py`
- Create: `examples/repopilot/evaluation/manifest.yaml`
- Create: `examples/repopilot/evaluation/tasks/<task-name>/...` for ten committed buggy repositories
- Create: `tests/test_repopilot/test_evaluation.py`
- Modify: `tests/test_repopilot/test_benchmark.py`
- Modify: `tests/test_repopilot/test_cli.py`
- Modify: `tests/test_repopilot/test_examples.py`

**Interfaces:**
- Produces `EvaluationStrategy`, `EvaluationCaseResult`, `EvaluationReport`,
  `EvaluationRunner.run(manifest, strategies, repetitions=1)`.

- [ ] **Step 1: Write evaluation aggregation tests**

Assert success rate, failure distribution, median duration, tokens, estimated cost,
repair/replan counts, changed-file compliance, and separation of scripted from model
quality results.

- [ ] **Step 2: Implement strategy runner and report writers**

Write timestamped JSON and Markdown reports. Preserve every run ID and failed result.
Support `scripted`, `model_no_retrieval`, and `model_with_retrieval`.

- [ ] **Step 3: Add ten independently reproducible fixtures**

Create tasks for boundary, exception, branching, normalization, collection/import,
multi-file flow, mutation, regression, replan, and scope enforcement. Every fixture
is a small Git repository with failing target test, optional regression test, manifest,
and allowed paths. Golden expected behavior belongs in tests, not model-visible
instructions.

- [ ] **Step 4: Add fixture integrity tests**

For every fixture, initialize/copy safely, assert the baseline target fails, assert
manifest paths resolve, and assert no generated run artifacts are included.

- [ ] **Step 5: Add CLI**

Add `openh repopilot evaluate <manifest> --strategy ... --repetitions 1`. Default to
one repetition and require explicit confirmation/config for larger live matrices.

- [ ] **Step 6: Run deterministic evaluation tests**

Run: `python -m pytest -q tests/test_repopilot/test_evaluation.py tests/test_repopilot/test_benchmark.py tests/test_repopilot/test_examples.py tests/test_repopilot/test_cli.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/openharness/repopilot/evaluation.py src/openharness/repopilot/benchmark.py src/openharness/repopilot/cli.py examples/repopilot/evaluation tests/test_repopilot
git commit -m "feat(repopilot): add reproducible repair evaluation"
```

### Task 9: Shared Application Service and FastAPI

**Files:**
- Create: `src/openharness/repopilot/service.py`
- Create: `src/openharness/repopilot/api.py`
- Modify: `src/openharness/repopilot/cli.py`
- Modify: `pyproject.toml`
- Create: `tests/test_repopilot/test_service.py`
- Create: `tests/test_repopilot/test_api.py`

**Interfaces:**
- Produces `RepoPilotService.start/get/resume/cancel/events/artifacts/cleanup` and
  `create_app(service: RepoPilotService | None = None) -> FastAPI`.

- [ ] **Step 1: Write service tests**

Use a fake scheduler/store. Assert CLI-independent start/status/events/artifacts,
cooperative cancellation, unknown-run errors, and safe artifact names.

- [ ] **Step 2: Implement application service**

Own the in-process task registry and delegate durable state to RunStore. A restart can
read completed/persisted runs but does not claim distributed task recovery.

- [ ] **Step 3: Add FastAPI optional dependencies**

Add an `api` optional dependency group containing compatible FastAPI and Uvicorn
ranges. Keep core CLI installation working without importing FastAPI.

- [ ] **Step 4: Write API contract tests**

Use `TestClient` with the fake service. Assert 202 start, 200 status/events/artifacts,
404 unknown run, 409 invalid transition, 422 invalid manifest, and cancellation.

- [ ] **Step 5: Implement thin routes**

Routes validate and translate HTTP only. They do not instantiate phase handlers or
contain transition policy.

- [ ] **Step 6: Make CLI delegate to RepoPilotService**

Keep command output compatible. Add `openh repopilot serve --host 127.0.0.1 --port
8000`; refuse non-loopback host unless an explicit unsafe-local-demo flag is passed.

- [ ] **Step 7: Run tests**

Run: `python -m pytest -q tests/test_repopilot/test_service.py tests/test_repopilot/test_api.py tests/test_repopilot/test_cli.py`

Expected: PASS when API extras are installed; core tests skip with an explicit reason
when they are not.

- [ ] **Step 8: Commit**

```powershell
git add src/openharness/repopilot/service.py src/openharness/repopilot/api.py src/openharness/repopilot/cli.py pyproject.toml tests/test_repopilot
git commit -m "feat(repopilot): expose application service and API"
```

### Task 10: Read-Only Repository Insight Workflow

**Files:**
- Create: `src/openharness/repopilot/insight.py`
- Modify: `src/openharness/repopilot/models.py`
- Modify: `src/openharness/repopilot/cli.py`
- Create: `tests/test_repopilot/test_insight.py`

**Interfaces:**
- Produces `InsightRequest`, `InsightFinding`, `InsightReport`,
  `RepositoryInsightWorkflow.run(request)`.

- [ ] **Step 1: Write a deterministic insight-workflow test**

Use `ScriptedPhaseExecutor`. Assert SCAN -> RETRIEVE -> ANALYZE -> REPORT, citations
resolve inside the repository, no write tools are registered, and the output records
context/prompt versions.

- [ ] **Step 2: Implement read-only handlers and workflow definition**

Reuse WorkflowRuntime, RepositoryIndex, ContextBuilder, RunStore, and typed events.
Reject any executor tool request outside read/list/search.

- [ ] **Step 3: Add CLI and service entry point**

Add `openh repopilot insight <repo> --question <text>` and the equivalent service
method. Store a Markdown and JSON report.

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/test_repopilot/test_insight.py tests/test_repopilot/test_workflow.py tests/test_repopilot/test_cli.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/openharness/repopilot/insight.py src/openharness/repopilot/models.py src/openharness/repopilot/cli.py tests/test_repopilot/test_insight.py
git commit -m "feat(repopilot): add repository insight workflow"
```

### Task 11: Full Verification and Real DeepSeek Evaluation

**Files:**
- Modify only files implicated by verified failures.
- Generate ignored artifacts under task/run report directories.

**Interfaces:**
- Consumes all previous tasks.
- Produces real run/evaluation reports; no source interface.

- [ ] **Step 1: Run formatting and lint**

Run: `python -m ruff format --check src/openharness/repopilot tests/test_repopilot`

Run: `python -m ruff check src/openharness/repopilot tests/test_repopilot`

Expected: PASS.

- [ ] **Step 2: Run all RepoPilot tests in a unique workspace basetemp**

Run:

```powershell
$base = Join-Path (Resolve-Path .) ('.tmp/pytest-' + [guid]::NewGuid().ToString('N'))
python -m pytest -q tests/test_repopilot --basetemp $base
```

Expected: PASS.

- [ ] **Step 3: Run selected OpenHarness integration regressions**

Run:

```powershell
python -m pytest -q tests/test_engine/test_query_engine.py::test_query_engine_executes_tool_calls tests/test_commands/test_cli.py
```

Expected: PASS.

- [ ] **Step 4: Run the full suite with a generous timeout and saved log**

Run `python -m pytest -q --basetemp <unique-workspace-path>` and retain the complete
output. If upstream unrelated tests fail, classify and document them; do not claim a
full pass.

- [ ] **Step 5: Run one real DeepSeek smoke repair**

Use the configured OpenAI-compatible provider without printing environment variables
or credentials. Assert final COMPLETE/verified, worktree diff, required checks, typed
events, detailed token usage when returned, and cost estimate.

- [ ] **Step 6: Run the declared 10 x 3 x 1 evaluation**

Run scripted first, then `model_no_retrieval`, then `model_with_retrieval`. Preserve
failures and stop if provider balance/rate-limit errors make results incomparable.

- [ ] **Step 7: Diagnose and fix verified failures test-first**

For each failure, reproduce with the smallest command, identify root cause, add or
tighten a regression test, implement the minimal fix, rerun focused and affected
suites, and commit by subsystem.

### Task 12: Portfolio Documentation and Final Delivery

**Files:**
- Modify: `README.md`
- Replace/fix: `docs/repopilot.md`
- Create: `docs/repopilot-architecture.md`
- Create: `docs/repopilot-evaluation.md`
- Modify: `examples/repopilot/README.md`

**Interfaces:**
- Consumes verified implementation and real reports.
- Produces user-facing reproducible evidence only.

- [ ] **Step 1: Fix encoding and write an accurate beginner-to-engineer guide**

Explain OpenHarness versus RepoPilot, one complete run, workflow/recovery/verification,
retrieval, events, worktrees, safety boundaries, and exact PowerShell commands.

- [ ] **Step 2: Add architecture and sequence diagrams**

Use Mermaid source showing application, domain, OpenHarness, and external layers plus
the model/tool/observation/verification loop.

- [ ] **Step 3: Publish evaluation methodology and measured results**

Include task taxonomy, strategies, provider/model/prompt version, repetitions,
metrics, failures, limitations, and links to generated JSON/Markdown reports. Include
only values produced by Task 11.

- [ ] **Step 5: Verify every documented command**

Execute quickstart, show/report, deterministic evaluation, API startup/health, and
insight commands in PowerShell. Correct documentation rather than hiding deviations.

- [ ] **Step 6: Run final diff and secret review**

Run:

```powershell
git diff --check
git status --short
git grep -n -I -E 'sk-[A-Za-z0-9_-]{12,}|DEEPSEEK_API_KEY=.+'
```

Expected: no whitespace errors, no unintended generated worktrees staged, and no
credentials.

- [ ] **Step 7: Commit documentation**

```powershell
git add README.md docs examples/repopilot/README.md
git commit -m "docs(repopilot): publish architecture and evaluation evidence"
```

- [ ] **Step 8: Review commit range and push**

Review `git diff origin/feat/repopilot-scheduler...HEAD`, rerun the final focused
verification, and push only the feature branch to `origin`.
