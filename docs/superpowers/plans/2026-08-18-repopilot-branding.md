# RepoPilot Branding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the public fork to RepoPilot and make its GitHub and README metadata discoverable without obscuring OpenHarness attribution.

**Architecture:** Keep code and package names unchanged to preserve compatibility. Change only the public repository identity, README first screen, Git remote URL, repository description, and topics; keep the upstream remote and licensing intact.

**Tech Stack:** Git, GitHub repository settings, Markdown

## Global Constraints

- Public repository name is exactly `RepoPilot`.
- README title is `RepoPilot — 基于 OpenHarness 开发的代码修复 Agent`.
- Preserve `HKUDS/OpenHarness` attribution, MIT license, and the `upstream` remote.
- Preserve the honest limitations around the 45-task evaluation.
- Do not claim guaranteed GitHub search rank or instant indexing.

---

### Task 1: Make the README RepoPilot-first

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`

**Interfaces:**
- Consumes: existing RepoPilot architecture and evaluation documentation
- Produces: a first-screen project identity containing the exact searchable name and upstream attribution

- [ ] **Step 1: Replace the root heading in both README files**

Use `RepoPilot` as the H1 and place `基于 OpenHarness 开发的代码修复 Agent` directly beneath it.

- [ ] **Step 2: Add explicit upstream attribution and portfolio links**

Link `HKUDS/OpenHarness`, `docs/repopilot-architecture.md`, and `docs/repopilot-evaluation.md` near the top without removing existing measured results or limitations.

- [ ] **Step 3: Verify Markdown changes**

Run: `git diff --check`

Expected: exit code 0 with no whitespace errors.

- [ ] **Step 4: Commit the documentation change**

```bash
git add README.md README.zh-CN.md docs/superpowers/plans/2026-08-18-repopilot-branding.md
git commit -m "docs(repopilot): make repository identity discoverable"
```

### Task 2: Publish the documentation branch

**Files:**
- No file changes

**Interfaces:**
- Consumes: committed branding documentation
- Produces: a pushed branch and updated default branch

- [ ] **Step 1: Push the current feature branch**

Run: `git push -u origin feat/repopilot-swebench-calibration`

Expected: the remote branch includes the branding commits.

- [ ] **Step 2: Merge the committed branding changes into `main`**

Use a clean checkout of `main`, merge `feat/repopilot-swebench-calibration`, and push `main` without rewriting history.

- [ ] **Step 3: Verify remote main contains the README title**

Fetch `README.md` from the public repository and confirm it contains `RepoPilot` and `HKUDS/OpenHarness`.

### Task 3: Rename and classify the GitHub repository

**Files:**
- No local file changes

**Interfaces:**
- Consumes: GitHub admin access to `w2528298635-eng/OpenHarness`
- Produces: `w2528298635-eng/RepoPilot` with searchable metadata

- [ ] **Step 1: Rename the repository in GitHub Settings**

Change repository name from `OpenHarness` to `RepoPilot`.

- [ ] **Step 2: Set repository description**

Set description to: `RepoPilot: an OpenHarness-based code repair agent with deterministic scheduling, code RAG, Cross-Encoder reranking, recovery, and SWE-bench evaluation.`

- [ ] **Step 3: Set repository topics**

Set: `repopilot`, `openharness`, `coding-agent`, `code-repair`, `rag`, `reranker`, `swe-bench`, `llm-agent`.

### Task 4: Synchronize and verify local and public identity

**Files:**
- Modify: shared local Git remote configuration only

**Interfaces:**
- Consumes: renamed public repository
- Produces: working local fetch/push configuration and verified public URLs

- [ ] **Step 1: Update local origin**

Run: `git remote set-url origin https://github.com/w2528298635-eng/RepoPilot.git`

- [ ] **Step 2: Verify remotes**

Run: `git remote -v`

Expected: `origin` points to `w2528298635-eng/RepoPilot`; `upstream` remains `HKUDS/OpenHarness`.

- [ ] **Step 3: Verify new and old public URLs**

Confirm `https://github.com/w2528298635-eng/RepoPilot` loads and the old `OpenHarness` URL redirects.

- [ ] **Step 4: Verify discoverability metadata**

Confirm the public page shows the RepoPilot repository name, description, topics, README heading, and upstream attribution. Record that GitHub search indexing may update asynchronously.
