# RepoPilot Platform Upgrade Design

Date: 2026-07-30

## 1. Goal

Upgrade RepoPilot from a working code-repair prototype into a resume-ready Agent
engineering project with reproducible evidence. The upgraded project must show that
the author understands how to separate probabilistic model decisions from
deterministic workflow control, verification, recovery, safety, observability, and
evaluation.

The target is not a general-purpose replacement for OpenHarness. OpenHarness remains
responsible for provider access, the model/tool loop, streaming events, base tools,
and low-level runtime assembly. RepoPilot adds the domain policy needed to repair
trusted local Python repositories reliably.

## 2. Success Criteria

The upgrade is complete only when all of the following are true:

1. Existing RepoPilot behavior is preserved through a smaller, reusable workflow
   runtime instead of a monolithic scheduler.
2. A run can be inspected, resumed, and explained through typed lifecycle events and
   a stable summary model.
3. Verification combines repository checks instead of trusting one pytest command.
4. Failures are classified and routed through explicit retry, repair, replan, or stop
   decisions with bounded budgets.
5. Worktree creation and cleanup behave predictably on Windows and do not pollute
   example fixtures.
6. Local context retrieval can select relevant repository evidence within a token or
   character budget and can be disabled for an ablation comparison.
7. At least ten distinct local bug tasks provide deterministic, reproducible
   evaluation inputs.
8. Three declared strategies can be compared without fabricating results:
   deterministic/scripted control, model without retrieval, and model with retrieval.
9. A small FastAPI layer exposes run creation, status, events, and results without
   duplicating scheduler logic.
10. A second read-only repository-insight workflow demonstrates reuse of the runtime.
11. Tests, reports, README instructions, architecture diagrams, resume bullets, and
    interview questions match the implemented behavior.

## 3. Scope and Non-Goals

### In scope

- Reliable workflow runtime and handler registry
- Composite verification and richer failure taxonomy
- Recovery policy and explicit terminal reasons
- Typed events, summaries, token/cost accounting, redaction, and bounded payloads
- Worktree lifecycle and Windows path handling
- Local lexical retrieval and context construction
- Versioned prompts and RepoPilot-facing provider seam for deterministic tests
- Ten-task local benchmark and strategy comparison
- FastAPI application layer
- Read-only repository-insight workflow
- Documentation, demo, evaluation evidence, resume bullets, and interview material

### Out of scope

- Training or fine-tuning a model
- Multi-agent teams
- Kubernetes, distributed queues, or production multi-tenant deployment
- Running untrusted code in a security sandbox
- Automatic dependency installation
- Automatic commit, pull request, merge, or push from a repair run
- Reimplementing OpenHarness QueryEngine, ToolRegistry, provider clients, or MCP
- Claiming benchmark improvements before real measurements exist
- Vector databases or embedding services unless lexical retrieval is measured and
  shown to be insufficient

## 4. Architecture

The upgraded system uses four layers:

```text
CLI / FastAPI
      |
Application services
      |
RepoPilot domain runtime
  WorkflowDefinition
  WorkflowRuntime
  PhaseHandler registry
  TransitionPolicy
  FailurePolicy
  CompositeVerifier
  ContextBuilder
  RunStore / EventSink
      |
OpenHarness infrastructure
  RuntimeBundle / QueryEngine
  ToolRegistry / BaseTool
  provider clients
  streaming events
  worktree helpers
      |
Git repository / Python / DeepSeek-compatible API
```

Dependencies point downward. The API and CLI call the same application service.
Phase handlers may call OpenHarness, but OpenHarness does not import RepoPilot.
Domain policy is testable without a live model by using scripted phase results.

## 5. Domain Models

Pydantic models form the contracts between components. They validate data at system
boundaries and make persisted run state forward-compatible.

Key models:

- `WorkflowDefinition`: workflow name, version, initial phase, terminal phases, and
  registered phase handlers.
- `RunContext`: immutable task configuration plus current mutable run state.
- `PhaseRequest` and `PhaseResult`: standardized input and output of a phase handler.
- `VerificationCheck` and `VerificationSuiteResult`: one check and the aggregate
  verdict.
- `FailureRecord`: category, signature, source phase, retryability, and artifact
  references.
- `RecoveryDecision`: next phase, reason, and budget effect.
- `RunEvent`: typed lifecycle event with schema version and bounded metadata.
- `RunSummary`: final phase, reason, duration, model/tool/verification counts, token
  usage, estimated provider cost, changed files, and artifact locations.

Existing serialized state remains readable. New fields receive defaults, and a
schema-version migration function handles incompatible future changes.

## 6. Workflow Runtime

`WorkflowRuntime` owns the generic execution loop:

1. Load or create `RunContext`.
2. Resolve the handler for the current phase.
3. Emit phase-started.
4. Execute the handler.
5. Persist the result and emit phase-finished.
6. Ask transition and failure policies for the next phase.
7. Atomically checkpoint state.
8. Stop only on a terminal phase, cancellation, or exhausted budget.

The runtime does not know how ANALYZE or EXECUTE works. Each `PhaseHandler` has one
responsibility and can be unit-tested independently. The code-repair workflow
registers PRECHECK, ANALYZE, PLAN, EXECUTE, VERIFY, REPAIR, and REPLAN handlers. The
repository-insight workflow registers read-only SCAN, RETRIEVE, ANALYZE, and REPORT
handlers.

The current `Scheduler` remains as a compatibility facade during migration and
delegates to the runtime. Once callers and tests use the new service, obsolete
internal code can be removed without changing the CLI contract.

## 7. Model and Tool Execution

Model-backed phase handlers reuse OpenHarness runtime assembly and QueryEngine.
RepoPilot supplies:

- phase-specific system and task prompts;
- validated structured context from previous phases;
- the phase's allowed tool set;
- the worktree path;
- event translation from OpenHarness events into RepoPilot events.

The OpenHarness tool registry remains the source of truth for tool schemas and
routing. RepoPilot adds phase-scoped policy and constrained wrappers only where the
domain requires narrower write boundaries.

Tool calls pass through parameter validation, path policy, execution, output
truncation/redaction, and observation recording. Raw oversized output is stored as an
artifact while the event contains a short summary and artifact reference.

## 8. Composite Verification

Verification is a sequence of independent checks with declared severity:

1. Worktree and diff-scope check
2. Python syntax/compile check for changed Python files
3. Target verification command
4. Optional broader regression command
5. Diff sanity check, including empty or suspicious test-only changes

Checks run without shell interpolation, use timeouts, capture bounded output, and
store full logs as artifacts. Required checks must pass for the suite to pass.
Optional checks remain visible but do not silently change the verdict.

Failure categories include configuration, baseline-not-reproducible, timeout,
syntax, collection, assertion, import/dependency, permission/path, no-diff,
out-of-scope-diff, repeated-failure, provider, structured-output, cancellation, and
internal errors.

## 9. Recovery and Budgets

`FailurePolicy` maps the current phase and normalized failure record to a
`RecoveryDecision`.

Examples:

- Structured output invalid once: retry the same read-only phase.
- Target tests fail after a real diff: REPAIR.
- Same failure signature or same diff repeats: REPLAN.
- No diff after EXECUTE: REPLAN.
- Permission or out-of-scope write: stop immediately.
- Provider/transient failure: bounded same-phase retry with backoff metadata.
- Syntax failure after edit: REPAIR.
- Baseline already passes: stop as invalid evaluation input.

All paths are bounded by phase calls, repair attempts, replan attempts, wall time,
token budget, changed-file budget, repeated actions, repeated diffs, and provider
retry budget. Terminal reasons are stable enum values rather than free-form strings.

## 10. Worktree Lifecycle

Runs operate on an isolated Git worktree. To support Windows:

- worktree roots use a short configurable base path;
- directory names use compact run identifiers;
- all resolved paths are validated against the intended root;
- cleanup uses Git worktree removal/prune rather than raw recursive deletion;
- active and completed runs have explicit retention policies;
- tests never copy live worktree directories;
- example-generated artifacts are ignored and discoverable through run metadata.

The default keeps failed worktrees for inspection and allows successful worktrees to
be cleaned explicitly or by a configured retention policy. No user worktree is
deleted merely because a test finishes.

## 11. Context Retrieval

The first retrieval implementation is deterministic and local:

1. Discover allowed text/code files while respecting ignore rules and size limits.
2. Parse Python symbols with `ast` where possible and fall back to bounded text
   chunks.
3. Index file paths, symbol names, imports, docstrings, and chunk text.
4. Rank candidates using lexical/BM25-style relevance plus path/symbol bonuses.
5. Add failure output and already identified files as high-priority evidence.
6. Assemble deduplicated context within a declared budget.

`ContextBuilder` records which chunks were selected and why. Retrieval can be
disabled so evaluation can compare equivalent prompts with and without retrieved
context. No claim that RAG improves results is made until the evaluation supports it.

## 12. Prompt and Provider Boundaries

Prompts are named, versioned templates with declared input variables and output
schema. Each run records the prompt version and provider/model identifier.

OpenHarness remains responsible for real provider clients. RepoPilot defines a small
phase-execution protocol so tests and deterministic evaluation can inject a
`ScriptedPhaseExecutor`. This is not a duplicate provider SDK; it is a test seam
around the model-backed phase handler.

Usage accounting stores input, output, cache-hit, and total tokens when the provider
returns them. Cost estimation uses a configurable price table stamped with a version
and is labeled as an estimate, never as the provider invoice.

## 13. Evaluation

The local suite contains at least ten small Git repositories covering distinct
repair behaviors, such as:

- boundary condition;
- exception behavior;
- wrong branching;
- parsing/normalization;
- collection/import failure;
- multi-file data flow;
- state mutation;
- regression outside the target test;
- misleading initial hypothesis requiring replan;
- change-scope enforcement.

Every task has:

- a committed buggy baseline;
- a task manifest;
- a failing target test;
- optional regression tests;
- allowed paths;
- deterministic validation;
- documented expected behavior without exposing a golden patch to the model.

Strategies:

- `scripted`: validates orchestration deterministically and is not presented as model
  quality.
- `model_no_retrieval`: real provider with retrieval disabled.
- `model_with_retrieval`: same provider settings with retrieval enabled.

The primary first pass is 10 tasks x 3 strategies x 1 run. Only unstable or
high-value model tasks receive additional repetitions. Reports include success rate,
terminal reason distribution, duration, phase/model calls, repair/replan counts,
tokens, estimated cost, and changed-file scope. Scripted results are displayed
separately from model success rates.

## 14. API and Second Workflow

FastAPI is a thin application adapter:

- `POST /runs`: validate a manifest and start a run;
- `GET /runs/{run_id}`: return status and summary;
- `GET /runs/{run_id}/events`: return paginated typed events;
- `GET /runs/{run_id}/artifacts`: list safe artifact metadata;
- `POST /runs/{run_id}/cancel`: request cooperative cancellation.

The initial server is single-process and intended for local demonstration. It does
not claim distributed durability.

The repository-insight workflow is read-only. It retrieves repository context and
produces an architecture/risk report with cited file paths. Its purpose is to prove
that the workflow runtime is reusable without adding multi-agent complexity.

## 15. Observability and Safety

Lifecycle events cover run, phase, model request, tool action, observation,
verification check, transition, checkpoint, recovery decision, cancellation, and
completion. Events have timestamps, correlation identifiers, schema versions, and
bounded/redacted payloads.

API keys and common secret patterns are never persisted. Tool output and exception
messages pass through redaction. Paths exposed by the API are constrained to known
run artifacts. The project continues to state clearly that a Git worktree is
isolation for source changes, not a sandbox for malicious code.

## 16. Testing Strategy

Testing proceeds from small deterministic units to real provider evaluation:

1. Domain-model validation and serialization tests
2. Pure transition, budget, and failure-policy tests
3. Phase-handler tests with scripted executors
4. Composite-verifier subprocess tests
5. RunStore/event migration, redaction, and failure-tolerance tests
6. Windows-oriented worktree path and cleanup tests
7. End-to-end local workflow tests in temporary Git repositories
8. CLI and FastAPI contract tests
9. Retrieval relevance and budget tests
10. Existing RepoPilot and selected OpenHarness regression tests
11. Real DeepSeek smoke run
12. Declared benchmark execution

Implementation follows test-driven slices. Each slice adds a failing test, the
smallest implementation, and regression verification before the next slice.

## 17. Delivery Order

### Milestone 1: Reliable Agent Runtime

Workflow/domain split, composite verification, failure/recovery policy, typed events
and summaries, usage accounting, and worktree lifecycle fixes.

### Milestone 2: Evaluation and Context

Ten tasks, strategy runner, deterministic scripted executor, lexical retrieval,
context builder, prompt versions, and evaluation reports.

### Milestone 3: Application and Portfolio

FastAPI, repository-insight workflow, README, diagrams, reproducible demo, real
evaluation results, resume bullets, and interview material.

Each milestone is committed separately and must pass its own focused tests. The final
push occurs only after the complete verification pass and a review of the committed
diff.

## 18. Evidence Rules

- Do not invent success rates, speedups, token savings, or benchmark counts.
- Generated reports must identify provider, model, prompt version, strategy, and run
  count.
- A failed evaluation remains in the evidence rather than being silently deleted.
- Documentation distinguishes deterministic orchestration tests from model-quality
  evaluation.
- Resume wording is generated only from features and measurements present in the
  final repository.
