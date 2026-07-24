# RepoPilot Scheduler Design

## 1. Purpose

RepoPilot is a local Python bug-fixing workflow built on OpenHarness. Its purpose is
to demonstrate a deterministic, observable scheduling layer around the existing
model-driven ReAct loop.

The first release accepts:

- a local Git repository;
- a natural-language bug description;
- an explicit `pytest` verification command;
- optional allowed-path and budget policies.

It produces:

- an isolated worktree containing the attempted fix;
- a verified success or a structured failure;
- structured analysis and repair plans;
- an append-only action/observation trace;
- verification logs, diffs, budget usage, and a final report.

RepoPilot does not let the model decide whether a repair succeeded. Only the
configured verifier can move a run to `COMPLETE`.

## 2. Scope

### 2.1 Included

- Local Git repositories.
- Python projects verified by an explicit `pytest` argv.
- Trusted local code execution without Docker.
- An outer Plan-Execute-Verify-Repair state machine.
- Inner OpenHarness `QueryEngine` ReAct loops for model-driven phases.
- Phase-specific prompts and tool allowlists.
- Worktree isolation.
- Structured Pydantic state and outputs.
- Atomic checkpoints and resume.
- Failure classification, bounded repair, and bounded replanning.
- Token, time, phase-call, changed-file, and repetition budgets.
- Deterministic tests with fake model clients.
- Optional paid end-to-end tests with a configured provider.
- A local benchmark followed by a small public-bug validation phase.

### 2.2 Excluded

- JavaScript, Java, or other language adapters.
- Docker or untrusted repository execution.
- Automatic dependency installation.
- GitHub Issue, pull request, CI, or merge automation.
- Multi-agent collaboration.
- Automatic discovery or generation of the verification command.
- Free-form shell verification strings.
- Web UI.
- Fine-tuning or RAG.
- Automatic commits to the user's original branch.
- Claimed benchmark improvements without measured evidence.

## 3. Architectural Decision

RepoPilot uses two scheduling layers:

1. `RepoPilotScheduler` is a deterministic outer workflow that selects phases and
   applies transition and budget policies.
2. OpenHarness `QueryEngine` and `run_query` remain the inner ReAct loop that lets
   a model use tools within one constrained phase.

This preserves OpenHarness API clients, messages, streaming, tools, permissions,
hooks, context management, token accounting, and worktree infrastructure. It
avoids modifying the monolithic core loop while still replacing unconstrained
end-to-end behavior with an explicit workflow.

The existing `autopilot` service is not the implementation base because it is
tightly coupled to GitHub, PR, CI, and release policies. RepoPilot may integrate
with it later through an adapter, but the local scheduler remains independent.

## 4. End-to-End Flow

```text
TaskSpec
  -> create isolated worktree
  -> PRECHECK
  -> ANALYZE
  -> PLAN
  -> EXECUTE
  -> VERIFY
      -> COMPLETE
      -> REPAIR -> VERIFY
      -> REPLAN -> EXECUTE
      -> FAILED
```

### 4.1 PRECHECK

Run the user-provided verification command against the untouched worktree.

- A normal test failure proves the bug is reproducible and moves to `ANALYZE`.
- A passing command terminates with `bug_not_reproduced`.
- A missing executable, timeout after one retry, or collection/infrastructure
  failure terminates with `invalid_verification_environment`.

The expected initial test failure is not treated as a scheduler failure.

### 4.2 ANALYZE

Run a read-only phase agent that locates the suspected root cause and emits a
validated `AnalysisResult`.

- The result must cite existing files and symbols.
- Evidence paths must resolve inside the worktree.
- Invalid structured output may be regenerated once.
- Repeated invalid output terminates with `invalid_analysis`.

### 4.3 PLAN

Generate a validated `RepairPlan` from the task and analysis.

- Each step names target files and expected behavior.
- Target paths must exist or be valid paths for an explicitly planned new file.
- The plan may not widen the user policy.
- Invalid output may be regenerated once.
- Repeated invalid output terminates with `invalid_plan`.

### 4.4 EXECUTE

Run an editing phase agent with the validated plan.

- Only scoped tools are exposed.
- The agent cannot change the verification command.
- A non-empty, policy-compliant diff moves to `VERIFY`.
- An empty diff requests `REPLAN`.
- A prohibited-path change terminates with `policy_violation`.
- A transient model error follows existing provider retry behavior; a repeated
  phase failure terminates.

### 4.5 VERIFY

Run the exact verification argv through `PythonPytestVerifier` with
`shell=False`.

- Exit code zero moves to `COMPLETE`.
- Test failures move to `REPAIR`.
- A collection/import failure introduced by the current diff moves to `REPAIR`.
- A missing executable or persistent infrastructure error terminates.
- A timeout is retried once if the global budget permits, then terminates.

Only this state can authorize `COMPLETE`.

### 4.6 REPAIR

Run an editing phase agent with the current plan, diff, latest verification
summary, prior failure signature, and remaining budget.

- A changed diff moves to `VERIFY`.
- No changed diff moves to `REPLAN`.
- Two identical failure signatures across repair attempts move to `REPLAN`.
- Exhausted repair budget terminates with `repair_budget_exhausted`.

### 4.7 REPLAN

Reanalyze the current modified worktree and produce a replacement hypothesis and
plan without discarding the current diff.

- A valid plan moves to `EXECUTE`.
- Exhausted replan budget terminates with `replan_budget_exhausted`.

## 5. Components

### 5.1 `RepoPilotScheduler`

Owns the current phase, invokes phase handlers, asks `TransitionPolicy` for the
next state, checks budgets before transitions, writes checkpoints, and emits
events. It does not read files, edit code, or run tests directly.

### 5.2 `PhaseAgentRunner`

Builds a fresh `QueryEngine` for each model-driven phase. It receives a
phase-specific prompt, scoped tool registry, working directory, and compact
structured context. It returns a phase result and usage snapshot.

Each phase uses a fresh conversation. Cross-phase continuity is provided by
validated state, not free-form message history.

### 5.3 `ScopedToolRegistry`

Provides a read-only filtered view of an existing `ToolRegistry`.

| Phase | Tools |
|---|---|
| `ANALYZE` | `read_file`, `glob`, `grep`, read-only LSP |
| `PLAN` | none |
| `EXECUTE` | `read_file`, `glob`, `grep`, `file_edit`, `file_write`, LSP |
| `VERIFY` | none; the scheduler invokes the verifier |
| `REPAIR` | same as `EXECUTE` |
| `REPLAN` | same as `ANALYZE` |

The general-purpose `bash` tool is not exposed in the first release.

### 5.4 `PythonPytestVerifier`

Executes a validated argv with `shell=False`, a configured timeout, and the
worktree as `cwd`. It captures exit code, stdout, stderr, duration, a summarized
failure category, and a stable failure signature.

It does not install dependencies or rewrite the command.

### 5.5 `TransitionPolicy`

Maps a phase result and current run state to a `TransitionDecision`. All
transitions are deterministic and unit-testable. Model prose is not parsed to
select a state.

### 5.6 `BudgetController`

Enforces:

- maximum phase-agent calls;
- maximum repair attempts;
- maximum replan attempts;
- wall-clock timeout;
- token limit when provider usage is available;
- maximum changed-file count;
- repeated action and repeated diff thresholds.

Budget exhaustion produces a terminal failure with the exact exhausted resource.

### 5.7 `RunStore`

Persists:

```text
.openharness/repopilot/runs/<run-id>/
  state.json
  events.jsonl
  analysis.json
  plan.json
  diff.patch
  verification-<attempt>.json
  verification-<attempt>.log
  report.md
```

State snapshots use temporary files followed by atomic replacement.
`events.jsonl` is append-only.

### 5.8 `WorkspaceManager`

Adapts existing OpenHarness worktree functionality. It creates an isolated branch,
captures diff snapshots and changed files, and leaves the worktree available for
inspection after success or failure. The first release does not push or merge.

## 6. Data Contracts

### 6.1 Task input

```python
class RepoTaskSpec(BaseModel):
    repo_path: Path
    issue: str
    verify_command: list[str]
    allowed_paths: list[str] | None = None
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
```

`verify_command` must be a non-empty argv. Shell metacharacter interpretation is
not supported because execution never uses a shell.

### 6.2 Analysis and plan

```python
class CodeEvidence(BaseModel):
    file: str
    symbol: str = ""
    line: int | None = None
    observation: str

class AnalysisResult(BaseModel):
    suspected_files: list[str]
    root_cause: str
    evidence: list[CodeEvidence]
    affected_symbols: list[str] = []
    confidence: float

class PlanStep(BaseModel):
    id: str
    description: str
    target_files: list[str]
    expected_behavior: str

class RepairPlan(BaseModel):
    hypothesis: str
    steps: list[PlanStep]
    expected_files: list[str]
    expected_behavior: str
```

### 6.3 Actions and observations

```python
class ActionRecord(BaseModel):
    action_id: str
    phase: Phase
    action_type: str
    parameters: dict[str, object]
    source: Literal["model", "scheduler"]
    timestamp: datetime

class ObservationRecord(BaseModel):
    action_id: str
    status: Literal["success", "failure", "blocked"]
    summary: str
    raw_artifact_path: str | None = None
    metadata: dict[str, object] = {}
    failure_signature: str | None = None
```

### 6.4 Run state

`RepoRunState` stores the run id, current phase, task, structured phase outputs,
verification history, changed files, attempt counters, usage, timestamps, and
terminal reason. It is the sole durable source of truth.

## 7. Phase Prompts and Context

Each prompt defines the phase goal, prohibited behavior, exact output schema, and
remaining budget.

- `ANALYZE` forbids edits and requires source evidence.
- `PLAN` has no tools and must produce a minimal, file-scoped plan.
- `EXECUTE` follows the plan, remains inside allowed paths, and cannot declare
  verification success.
- `REPAIR` only addresses the latest verifier evidence and must request
  replanning when the hypothesis is no longer credible.

Only these items cross phase boundaries:

- `RepoTaskSpec`;
- `AnalysisResult`;
- `RepairPlan`;
- latest `VerificationResult`;
- current diff summary;
- budget snapshot.

Large raw outputs are stored as artifacts and represented in prompts by compact
summaries and paths.

## 8. Recovery and Idempotency

Every scheduler action has a deterministic action id such as
`<run-id>:verify:<attempt>`.

- A completed action with a durable observation is not repeated during resume.
- An incomplete model phase is rerun from its last stable checkpoint.
- An interrupted execute phase compares the pre-phase diff snapshot with the
  current diff and resumes through `REPLAN`.
- A fully persisted verifier result may be reused; an incomplete verification is
  rerun because tests can have transient state.

No resume path silently marks an incomplete phase as complete.

## 9. CLI

```text
openh repopilot run <task.yaml>
openh repopilot show <run-id>
openh repopilot resume <run-id>
openh repopilot report <run-id>
openh repopilot benchmark <benchmark.yaml>
```

The run command prints phase events and the final run id. Show prints state and
budget usage. Resume continues from the last stable checkpoint. Report renders
the saved Markdown report. Benchmark compares strategies using a manifest.

## 10. Testing

### 10.1 Unit tests

- Pydantic model validation.
- Every transition branch.
- Budget exhaustion.
- Failure-signature normalization.
- Repeated action and diff detection.
- Scoped tool registries.
- Verifier exit, timeout, and missing-executable handling.
- Atomic store writes and recovery.
- Verification command validation.

### 10.2 Integration tests

Temporary Git repositories and fake model clients cover:

- successful repair;
- one failed verification followed by repair;
- bug not reproduced;
- invalid model JSON;
- forbidden path modification;
- verifier timeout and infrastructure error;
- repair and replan exhaustion;
- process interruption and resume.

### 10.3 Paid end-to-end tests

Provider-backed tests are opt-in and excluded from default CI. They exercise a
small local bug task with the configured provider and confirm a real tool trace,
diff, verifier result, and report.

## 11. Evaluation

Development begins with 10–20 small reproducible Python bugs representing boundary
conditions, error handling, type conversion, state leakage, and regressions.

Each task has:

- a fixed initial commit;
- a bug description;
- the verification argv available to RepoPilot;
- an evaluation-only regression command not exposed to the Agent.

The same model, task, initial commit, verification, and budget are used for:

- baseline: native OpenHarness ReAct;
- candidate: RepoPilot scheduler.

Reports include:

- verified completion rate;
- evaluation-only regression pass rate;
- model phase calls;
- token usage when available;
- wall-clock time;
- repair recovery rate;
- repeated action count;
- policy violations;
- failure category distribution.

No improvement is claimed until these runs are executed. After local validation,
a small set of public real-world Python bugs is used as an external check.

## 12. Acceptance Criteria

The first release is complete when:

1. CLI task loading, run, show, resume, and report work on Windows.
2. PRECHECK prevents a passing baseline from being treated as a repair task.
3. Model-driven phases use fresh `QueryEngine` instances and scoped tools.
4. Only the verifier can authorize `COMPLETE`.
5. Repair and replan follow bounded deterministic transitions.
6. A process interruption can resume from a durable checkpoint.
7. Success and failure both produce inspectable diffs, logs, state, events, usage,
   and a Markdown report.
8. Default unit and integration tests do not require an API key.
9. At least one optional real-provider task completes with a captured trace.
10. Baseline and RepoPilot can run against the same local benchmark manifest
    without fabricated metrics.
