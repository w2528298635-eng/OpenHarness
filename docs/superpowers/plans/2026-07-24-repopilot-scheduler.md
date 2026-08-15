# RepoPilot Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, resumable Python bug-fixing scheduler around OpenHarness whose deterministic verifier—not model prose—decides whether a repair is complete.

**Architecture:** `RepoPilotScheduler` owns an explicit PRECHECK→ANALYZE→PLAN→EXECUTE→VERIFY→REPAIR/REPLAN state machine. Model-driven phases run through fresh, tool-scoped OpenHarness runtimes, while verification, transitions, budgets, persistence, worktree isolation, reporting, and benchmarking remain deterministic and independently testable.

**Tech Stack:** Python 3.11, Pydantic v2, Typer, PyYAML, asyncio subprocesses, Git worktrees, pytest, existing OpenHarness `QueryEngine`, tool registry, runtime builder, and provider clients.

## Global Constraints

- Support local Git repositories and explicit Python `pytest` argv only.
- Run trusted local code without Docker and never install dependencies automatically.
- Execute verification with `shell=False`; accept only `pytest ...`, `py.test ...`, or `<python> -m pytest ...`.
- Do not expose the general-purpose `bash` tool to RepoPilot model phases.
- Keep model phases fresh and pass only validated compact state between phases.
- Only a successful `VERIFY` result may authorize `COMPLETE`.
- Persist artifacts under `<original-repo>/.openharness/repopilot/runs/<run-id>/`.
- Leave isolated worktrees available after both success and failure; never push, merge, or modify the original branch.
- Default tests must use fake phase runners and require no API key.
- Do not claim benchmark improvements unless measured by the implemented benchmark command.

---

## File Structure

- `src/openharness/repopilot/models.py`: Pydantic input, phase output, event, verification, budget, transition, and durable state contracts.
- `src/openharness/repopilot/task_loader.py`: YAML loading, path normalization, and safe verification-command validation.
- `src/openharness/repopilot/store.py`: atomic state/artifact persistence and append-only event storage.
- `src/openharness/repopilot/verifier.py`: safe pytest subprocess execution, failure classification, and stable signatures.
- `src/openharness/repopilot/policy.py`: deterministic phase transitions and budget checks.
- `src/openharness/repopilot/tools.py`: phase-specific filtered tool registry.
- `src/openharness/repopilot/workspace.py`: Git worktree lifecycle, diff snapshots, and changed-file policy.
- `src/openharness/repopilot/prompts.py`: compact phase prompts and strict JSON contracts.
- `src/openharness/repopilot/phase_runner.py`: injectable phase-runner protocol and OpenHarness implementation.
- `src/openharness/repopilot/scheduler.py`: orchestration, checkpointing, recovery, and action/observation recording.
- `src/openharness/repopilot/report.py`: Markdown report renderer.
- `src/openharness/repopilot/benchmark.py`: manifest-driven local runs and measured summary output.
- `src/openharness/repopilot/cli.py`: `run`, `show`, `resume`, `report`, and `benchmark` commands.
- `src/openharness/cli.py`: register the RepoPilot Typer application.
- `tests/test_repopilot/`: focused unit and integration tests.
- `examples/repopilot/`: runnable task and benchmark examples.
- `docs/repopilot.md`: beginner-oriented operation and architecture guide.

### Task 1: Durable Data Contracts and Task Loading

**Files:**
- Create: `src/openharness/repopilot/__init__.py`
- Create: `src/openharness/repopilot/models.py`
- Create: `src/openharness/repopilot/task_loader.py`
- Create: `tests/test_repopilot/test_models.py`
- Create: `tests/test_repopilot/test_task_loader.py`

**Interfaces:**
- Produces: `Phase`, `BudgetConfig`, `RepoTaskSpec`, `CodeEvidence`, `AnalysisResult`, `PlanStep`, `RepairPlan`, `ActionRecord`, `ObservationRecord`, `VerificationResult`, `TransitionDecision`, `RepoRunState`.
- Produces: `load_task(path: Path) -> RepoTaskSpec` and `validate_verify_command(argv: list[str]) -> list[str]`.

- [ ] Write tests proving model defaults, state JSON round-trips, confidence bounds, non-empty issue validation, and command acceptance/rejection.
- [ ] Run `python -m pytest tests/test_repopilot/test_models.py tests/test_repopilot/test_task_loader.py -q` and confirm collection fails because `openharness.repopilot` does not exist.
- [ ] Implement strict Pydantic contracts, enums, UTC timestamp factories, YAML loading, repository resolution, Git-directory validation, and command validation without invoking a shell.
- [ ] Re-run the two test files and confirm all tests pass.
- [ ] Commit with `git commit -m "feat(repopilot): add task and state contracts"`.

### Task 2: Atomic Run Store and Event Trace

**Files:**
- Create: `src/openharness/repopilot/store.py`
- Create: `tests/test_repopilot/test_store.py`

**Interfaces:**
- Consumes: `RepoRunState`, `ActionRecord`, `ObservationRecord`.
- Produces: `RunStore.create(state)`, `save_state(state)`, `load_state(run_id)`, `append_event(event)`, `write_json(name, value)`, `write_text(name, text)`, `find_run(run_id, repo_path=None)`.

- [ ] Write tests proving atomic `state.json` replacement, JSONL append order, artifact creation, and recovery after an orphaned temporary state file.
- [ ] Run `python -m pytest tests/test_repopilot/test_store.py -q` and confirm failure due to missing `RunStore`.
- [ ] Implement repository-rooted run directories, UTF-8 JSON serialization, same-directory temporary writes plus `Path.replace`, and append-only events.
- [ ] Re-run the store tests and confirm all pass.
- [ ] Commit with `git commit -m "feat(repopilot): persist resumable run artifacts"`.

### Task 3: Safe Pytest Verifier

**Files:**
- Create: `src/openharness/repopilot/verifier.py`
- Create: `tests/test_repopilot/test_verifier.py`

**Interfaces:**
- Consumes: validated `list[str]`, worktree `Path`, timeout seconds, attempt number.
- Produces: `PythonPytestVerifier.verify(...) -> VerificationResult`, `normalize_failure_signature(stdout, stderr) -> str`.

- [ ] Write real subprocess tests for pass, assertion failure, collection/import failure, missing executable, timeout, and signature normalization that removes volatile paths, times, and addresses.
- [ ] Run `python -m pytest tests/test_repopilot/test_verifier.py -q` and confirm failure because the verifier is missing.
- [ ] Implement `asyncio.create_subprocess_exec(*argv, cwd=..., stdout=PIPE, stderr=PIPE)` with timeout termination, category classification, duration, captured output, and SHA-256 stable signatures.
- [ ] Re-run verifier tests and confirm all pass on Windows.
- [ ] Commit with `git commit -m "feat(repopilot): add deterministic pytest verifier"`.

### Task 4: Transition and Budget Policies

**Files:**
- Create: `src/openharness/repopilot/policy.py`
- Create: `tests/test_repopilot/test_policy.py`

**Interfaces:**
- Consumes: `RepoRunState`, phase outcome facts, `VerificationResult`.
- Produces: `TransitionPolicy.after_precheck`, `after_execute`, `after_verify`, `after_repair`, `after_replan`; `BudgetController.check(state, changed_files, now)`.

- [ ] Write table-driven tests for every legal transition, verifier-only completion, repeated failure→REPLAN, empty diff→REPLAN, terminal infrastructure errors, and each configured budget.
- [ ] Run `python -m pytest tests/test_repopilot/test_policy.py -q` and confirm failure because policy classes are missing.
- [ ] Implement pure deterministic policy methods returning `TransitionDecision(next_phase, terminal_reason, detail)` and exact budget failure reasons.
- [ ] Re-run policy tests and confirm all pass.
- [ ] Commit with `git commit -m "feat(repopilot): enforce transitions and budgets"`.

### Task 5: Scoped Tools and Workspace Isolation

**Files:**
- Create: `src/openharness/repopilot/tools.py`
- Create: `src/openharness/repopilot/workspace.py`
- Create: `tests/test_repopilot/test_tools.py`
- Create: `tests/test_repopilot/test_workspace.py`

**Interfaces:**
- Consumes: existing `ToolRegistry` and `WorktreeManager`.
- Produces: `ScopedToolRegistry.from_registry(registry, phase)`, `WorkspaceManager.create`, `diff`, `changed_files`, `validate_changed_files`, and `diff_signature`.

- [ ] Write tests proving exact phase allowlists, absent `bash`, source-registry immutability, isolated worktree creation, original-repo preservation, diff capture, and glob-based allowed-path enforcement.
- [ ] Run the two test files and confirm they fail because the adapters are missing.
- [ ] Implement a filtered registry using existing tool objects and a workspace adapter that calls Git using argv, captures porcelain/diff output, rejects paths escaping the worktree, and excludes `.git` plus RepoPilot artifacts.
- [ ] Re-run the tests and confirm all pass.
- [ ] Commit with `git commit -m "feat(repopilot): scope tools and isolate workspaces"`.

### Task 6: Phase Prompts and OpenHarness Runner

**Files:**
- Create: `src/openharness/repopilot/prompts.py`
- Create: `src/openharness/repopilot/phase_runner.py`
- Create: `tests/test_repopilot/test_prompts.py`
- Create: `tests/test_repopilot/test_phase_runner.py`

**Interfaces:**
- Produces: `build_phase_prompt(phase, state, diff_summary="") -> str`.
- Produces: `PhaseAgentRunner` protocol and `OpenHarnessPhaseRunner.run(phase, state, cwd) -> PhaseRunResult`.
- Accepts dependency injection for a runtime factory so tests can verify a fresh runtime and scoped tools without a provider.

- [ ] Write tests that each prompt states its goal, prohibitions, schema, budget, and compact prior evidence; verify separate calls receive separate runtime instances and invalid JSON gets exactly one regeneration attempt.
- [ ] Run the prompt/runner tests and confirm missing-symbol failures.
- [ ] Implement phase prompt rendering, fenced/prose JSON extraction, Pydantic validation, one schema-repair retry, fresh runtime creation, event consumption, and provider-usage extraction.
- [ ] Re-run tests and confirm all pass.
- [ ] Commit with `git commit -m "feat(repopilot): run constrained model phases"`.

### Task 7: Scheduler, Resume, and Reports

**Files:**
- Create: `src/openharness/repopilot/scheduler.py`
- Create: `src/openharness/repopilot/report.py`
- Create: `tests/test_repopilot/test_scheduler.py`
- Create: `tests/test_repopilot/test_report.py`

**Interfaces:**
- Consumes: task, workspace, store, verifier, phase runner, transition policy, and budget controller.
- Produces: `RepoPilotScheduler.start(task) -> RepoRunState`, `resume(run_id) -> RepoRunState`, and `render_report(state, run_dir) -> str`.

- [ ] Write fake-runner integration tests for successful repair, failed verification then repair, bug-not-reproduced, invalid model output, prohibited changes, timeout/infrastructure failure, exhausted repair/replan budgets, and interruption/resume without replaying durable actions.
- [ ] Run scheduler/report tests and confirm they fail because orchestration is absent.
- [ ] Implement idempotent action ids, durable observations, phase checkpointing, exact outer-loop transitions, artifact writes, diff/verification capture, terminal report generation, and resume rules from the design.
- [ ] Re-run scheduler/report tests and confirm all pass.
- [ ] Commit with `git commit -m "feat(repopilot): orchestrate resumable repairs"`.

### Task 8: CLI and Local Benchmark

**Files:**
- Create: `src/openharness/repopilot/cli.py`
- Create: `src/openharness/repopilot/benchmark.py`
- Modify: `src/openharness/cli.py`
- Create: `tests/test_repopilot/test_cli.py`
- Create: `tests/test_repopilot/test_benchmark.py`

**Interfaces:**
- Produces Typer commands `openh repopilot run|show|resume|report|benchmark`.
- Produces benchmark manifest models and JSON/Markdown aggregate results containing measured values or explicit `unavailable`.

- [ ] Write `CliRunner` tests for help, invalid task feedback, run id output, show, resume, report, benchmark manifest validation, and no fabricated comparison values.
- [ ] Run CLI/benchmark tests and confirm the `repopilot` command is absent.
- [ ] Implement a RepoPilot sub-application, runtime dependency construction, concise phase output, run lookup, report display, sequential manifest execution, and evidence-based aggregation.
- [ ] Re-run CLI/benchmark tests and confirm all pass.
- [ ] Commit with `git commit -m "feat(repopilot): expose CLI and benchmark workflow"`.

### Task 9: Runnable Example and Beginner Documentation

**Files:**
- Create: `examples/repopilot/discount_bug/discount.py`
- Create: `examples/repopilot/discount_bug/test_discount.py`
- Create: `examples/repopilot/task.example.yaml`
- Create: `examples/repopilot/benchmark.example.yaml`
- Create: `docs/repopilot.md`
- Modify: `README.md`
- Create: `tests/test_repopilot/test_examples.py`

**Interfaces:**
- Produces a deliberately failing local Python example, valid manifests, exact Windows/PowerShell commands, artifact explanations, and architecture/interview talking points.

- [ ] Write a test that copies the example into a temporary Git repository, proves its baseline test fails, and proves both YAML files load.
- [ ] Run `python -m pytest tests/test_repopilot/test_examples.py -q` and confirm it fails because examples are absent.
- [ ] Add the smallest discount-boundary bug, its regression test, manifests using argv arrays, a README entry, and a beginner guide explaining state transitions, Action/Observation traces, safety, run/resume/report, optional provider configuration, and limitations.
- [ ] Re-run the example test and confirm it passes.
- [ ] Commit with `git commit -m "docs(repopilot): add runnable example and guide"`.

### Task 10: Full Verification and Optional Provider Smoke Test

**Files:**
- Modify only if a failing test exposes a defect; every defect first receives a reproducing test in `tests/test_repopilot/`.

**Interfaces:**
- Verifies the finished feature against focused tests, relevant OpenHarness regression tests, CLI behavior, the example, and an opt-in real provider.

- [ ] Run `python -m pytest tests/test_repopilot -q` and record the exact pass/skip totals.
- [ ] Run `python -m pytest tests/test_engine/test_query_engine.py::test_query_engine_executes_tool_calls tests/test_commands/test_cli.py -q` and record regression results.
- [ ] Run `openh repopilot --help`, `openh repopilot run examples/repopilot/task.example.yaml`, then `show` and `report` for the emitted run id.
- [ ] If `DEEPSEEK_API_KEY` is available, map it only in the process to `OPENHARNESS_OPENAI_API_KEY`, execute one provider-backed example, and confirm a real phase trace, diff, verification result, and report without printing the key; otherwise document the exact opt-in command and skip result.
- [ ] Run `git status --short`, inspect the final diff, and commit any test-driven corrections with `git commit -m "test(repopilot): verify end-to-end workflow"`.
