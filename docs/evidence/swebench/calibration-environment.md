# RepoPilot SWE-bench calibration environment

Observed on 2026-07-30. This is environment evidence, not a repair-rate
result.

## Frozen subsets

- Formal sample: 45 public SWE-bench Verified tasks (10 easy, 15 medium,
  20 hard), spanning 12 repositories.
- Formal manifest SHA-256:
  `9b12758c4b8967d92d07200addb94ac12abe9244b2a208917ca854493e75cde0`.
- Calibration subset: three tasks from the frozen 45, one per difficulty,
  spanning scikit-learn, matplotlib, and Django.
- Calibration manifest SHA-256:
  `238a62a8ba8e9ea79734c3a98b877d11606ecb7c8892f225549d42f6d223ee91`.

Neither manifest contains the gold patch, test patch, FAIL_TO_PASS,
PASS_TO_PASS, or derived gold labels.

## Measured downloads

- Hugging Face metadata cache: 1.99 MiB.
- Three selected commit caches plus editable worktrees: 175.77 MiB.
- All 45 formal task worktrees plus the shared 12-repository commit cache and
  the earlier pilot worktrees: 1.476 GiB.
- Verification: 45 expected worktrees, zero extra worktrees, and zero
  base-commit mismatches.
- Repositories for unselected tasks downloaded: zero.
- Docker images for unselected tasks built: zero.

The repository layer therefore does not justify a 120 GiB requirement. Docker
base, environment, and instance images remain the dominant unknown and must be
measured with the official three-task calibration.

## Host and Docker observations

- Windows AMD64, 16 logical CPUs, 15.9 GiB RAM.
- E: had 82.4 GiB free at inspection time.
- Docker Desktop 4.84.0 reported `Engine running`, using the WSL2 backend.
- Docker reported 12.51 GiB used.
- Docker's disk image was located at
  `C:\Users\Administrator\AppData\Local\Docker\wsl`, not E:.

The Codex sandbox was denied access to Docker's Windows named pipe even while
the Docker Desktop UI showed the engine running. Moving Docker's disk image to
E: would restart Docker and may interrupt existing containers, so no migration
was performed automatically.

## Current status

The paid three-task inference calibration completed on 2026-07-30. Its sealed
artifacts and an explicit failure taxonomy are recorded in
[pilot-v2-inference.md](pilot-v2-inference.md). Two candidate patches are
awaiting official evaluation; neither may be reported as resolved.

The host's Docker daemon and WSL2 checks now pass, but the installed Windows
Python cannot run the official SWE-bench harness because it depends on the
Unix-only `resource` module. A normal Linux WSL distribution has not yet been
installed. Consequently, this document still makes no repair-rate, RAG-uplift,
cost, or statistical-significance claim.
