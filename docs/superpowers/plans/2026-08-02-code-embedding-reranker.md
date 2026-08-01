# Code Embedding and Cross-Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the generic dense retriever with pinned CodeRankEmbed and add an optional cached Cross-Encoder reranking stage, then publish a complete 45-task quantitative comparison.

**Architecture:** Keep the existing independent lexical/dense recall and weighted fusion as candidate generation. Make dense model identity explicit, then rerank only the fused top candidates in an isolated worker so model dependencies remain outside the OpenHarness runtime.

**Tech Stack:** Python 3.11, Pydantic v2, sentence-transformers, Hugging Face Transformers, SQLite, NumPy, pytest, Typer.

## Global Constraints

- CodeRankEmbed revision is `3c4b60807d71f79b43f3c4363786d9493691f8b1`.
- BGE reranker revision is `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
- Model files and runtime caches remain on E: and are not committed.
- Gold labels are consumed only after ranking.
- Every behavior change follows red-green-refactor and every experiment arm uses a distinct configuration-locked checkpoint.

---

### Task 1: Configurable professional dense encoder

**Files:**
- Modify: `src/openharness/repopilot/embedding.py`
- Modify: `src/openharness/repopilot/embedding_worker.py`
- Test: `tests/test_repopilot/test_embedding.py`

**Interfaces:**
- Produces: `LocalEmbeddingEncoder(model, revision, query_prefix, max_seq_length)`.
- Produces worker payload fields `model`, `revision`, `query_prefix`, and model-specific `model_key`.

- [ ] Add a failing test asserting the default payload names the pinned CodeRankEmbed model, uses its code-search query prefix, represents path/symbol/kind/code, and does not reuse the legacy BGE model key.
- [ ] Run `pytest tests/test_repopilot/test_embedding.py -q` and confirm that the payload assertions fail.
- [ ] Implement configurable model identity and CodeRank input representation in `embedding.py`.
- [ ] Update the worker to pass `revision` and `trust_remote_code`, apply the supplied query prefix, and remove the hard-coded passage prefix.
- [ ] Run the embedding tests and commit the green change.

### Task 2: Cross-Encoder reranker boundary

**Files:**
- Create: `src/openharness/repopilot/reranker.py`
- Create: `src/openharness/repopilot/reranker_worker.py`
- Create: `tests/test_repopilot/test_reranker.py`

**Interfaces:**
- Produces: `LocalCrossEncoderReranker.rank(query: str, matches: list[ScoredChunk], top_k: int) -> list[ScoredChunk]`.
- Worker consumes candidate IDs/text and returns ranked IDs/scores plus cache statistics.

- [ ] Add failing tests for deterministic candidate-only ordering, top-k enforcement, cached worker payload identity, malformed output, and surfaced worker stderr.
- [ ] Run the new test module and confirm failures are caused by the missing reranker.
- [ ] Implement the subprocess boundary and model-independent code representation.
- [ ] Implement the worker with pinned Transformers sequence classification, SQLite pair-score caching, truncation, and stable tie-breaking.
- [ ] Run the reranker tests and commit the green change.

### Task 3: Retrieval and context integration

**Files:**
- Modify: `src/openharness/repopilot/retrieval.py`
- Modify: `src/openharness/repopilot/context.py`
- Modify: `src/openharness/repopilot/models.py`
- Modify: `src/openharness/repopilot/scheduler.py`
- Modify: `tests/test_repopilot/test_context.py`
- Modify: `tests/test_repopilot/test_models.py`

**Interfaces:**
- `ContextBuilder` consumes embedding and reranker configuration.
- `RetrievalConfig` persists `embedding_model`, `embedding_revision`, `embedding_max_seq_length`, `reranker_enabled`, `reranker_model`, `reranker_revision`, `reranker_candidate_k`, and `reranker_top_k`.

- [ ] Add failing tests showing an enabled reranker receives fused candidates, changes final order, cannot add candidates, and defaults remain safe when disabled.
- [ ] Run focused tests and confirm expected failures.
- [ ] Request enough fused candidates for reranking, invoke the reranker, and preserve provenance reasons.
- [ ] Thread validated configuration from task YAML through scheduler to ContextBuilder.
- [ ] Run focused context/model/scheduler tests and commit.

### Task 4: Resumable evaluation configuration and CLI

**Files:**
- Modify: `src/openharness/repopilot/swebench/localization_execution.py`
- Modify: `src/openharness/repopilot/swebench/cli.py`
- Modify: `tests/test_repopilot/test_swebench_localization_execution.py`
- Modify: `tests/test_repopilot/test_swebench_cli.py`

**Interfaces:**
- Localization checkpoints record all embedding/reranker settings and reject configuration mixing.
- CLI exposes `--embedding-model`, `--embedding-revision`, `--reranker/--no-reranker`, `--reranker-model`, `--reranker-revision`, `--reranker-candidate-k`, and `--reranker-top-k`.

- [ ] Add failing tests for configuration persistence, mismatch rejection, and CLI option forwarding.
- [ ] Run focused tests and confirm the missing fields/options fail.
- [ ] Implement configuration forwarding and strict evaluation failure semantics.
- [ ] Run focused evaluation tests and commit.

### Task 5: Real model smoke and 45-task experiments

**Files:**
- Modify ignored operational scripts under `.openharness-swebench/` with `apply_patch` only.
- Create final evidence under `docs/evidence/swebench/` after all runs finish.

**Interfaces:**
- Produces separate checkpoints for CodeRankEmbed-only and CodeRankEmbed+Cross-Encoder.
- Produces machine-readable and Markdown comparisons against the existing published baseline.

- [ ] Verify the E: runtime imports the pinned models and download each model into the E: cache.
- [ ] Run the three-task pilot for CodeRankEmbed-only and CodeRankEmbed+Cross-Encoder; inspect latency, cache behavior, and ranking integrity.
- [ ] Adjust only operational batch/max-length settings if required by measured memory or timeout failures, then freeze them.
- [ ] Run both 45-task arms to complete checkpoints without removed failures.
- [ ] Generate reports and independently verify published aggregates against raw checkpoints.

### Task 6: Documentation and completion verification

**Files:**
- Modify: `README.md`
- Modify: `docs/repopilot.md`
- Modify: `docs/repopilot-architecture.md`
- Create: `docs/evidence/swebench/code-embedding-reranker-45.md`
- Create: `docs/evidence/swebench/code-embedding-reranker-45-summary.json`

**Interfaces:**
- Documentation reports exact model revisions, licenses, configurations, metrics, latency, and interpretation limits.

- [ ] Update documentation from the measured results, including any negative result and the non-commercial/development-set boundary where applicable.
- [ ] Run all RepoPilot tests, Ruff, `git diff --check`, and a raw-checkpoint-to-published-evidence consistency script.
- [ ] Confirm the worktree is clean after intentional commits and push the feature branch.
