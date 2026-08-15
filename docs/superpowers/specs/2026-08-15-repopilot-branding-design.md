# RepoPilot Branding Design

## Goal

Make the project easy to find when an interviewer searches GitHub for
`RepoPilot`, while preserving clear attribution to the upstream OpenHarness
project.

## Public identity

- GitHub repository name: `RepoPilot`
- README title: `RepoPilot — 基于 OpenHarness 开发的代码修复 Agent`
- Repository description: identify RepoPilot as an OpenHarness-based code-repair
  Agent with deterministic scheduling, code RAG, Cross-Encoder reranking,
  recovery, and SWE-bench evaluation.
- Repository topics: `repopilot`, `openharness`, `coding-agent`, `code-repair`,
  `rag`, `reranker`, `swe-bench`, and `llm-agent`.

## Attribution boundary

The README must state near the top that RepoPilot is developed from the
`HKUDS/OpenHarness` open-source project. The upstream remote, MIT license,
copyright notices, upstream links, and OpenHarness documentation remain intact.
The branding change must not claim that RepoPilot is an official OpenHarness
release or erase the upstream project's authorship.

## Repository and local Git changes

Rename `w2528298635-eng/OpenHarness` to `w2528298635-eng/RepoPilot` on GitHub.
After the rename, update the shared local `origin` remote to
`https://github.com/w2528298635-eng/RepoPilot.git`; keep `upstream` pointing to
`https://github.com/HKUDS/OpenHarness.git`.

GitHub redirects the old repository URL, clone, fetch, and push requests after a
rename, but the local remote is updated explicitly to avoid relying on that
redirect.

## README presentation

The first screen of both English and Chinese README files should lead with
RepoPilot, briefly state its relationship to OpenHarness, link to the measured
evaluation and architecture, and retain the existing honest limitations around
the 45-task localization evaluation. Detailed upstream OpenHarness and ohmo
content remains available below the RepoPilot introduction.

## Verification

- Confirm the new public repository URL resolves.
- Confirm the old URL redirects to the new repository.
- Confirm repository description and topics contain `RepoPilot` metadata.
- Confirm local `origin` and `upstream` point to the intended repositories.
- Run `git diff --check` and inspect the rendered README links.
- Do not claim a guaranteed GitHub search rank; search indexing and ranking are
  controlled by GitHub and may take time to refresh.
