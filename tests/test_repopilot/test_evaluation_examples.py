from pathlib import Path

from openharness.repopilot.benchmark import load_benchmark


def test_evaluation_suite_has_ten_independent_cases() -> None:
    root = Path(__file__).parents[2]
    manifest = load_benchmark(root / "examples/repopilot/evaluation/manifest.yaml")

    assert len(manifest.cases) == 10
    assert len({case.id for case in manifest.cases}) == 10
    for case in manifest.cases:
        assert case.task.exists()
        case_root = case.task.parent
        assert (case_root / "repo" / "test_app.py").exists()
        assert (case_root / "fix.patch").exists()
        assert not any(
            part in {".git", ".openharness", ".pytest_cache"}
            for path in case_root.rglob("*")
            for part in path.relative_to(case_root).parts
        )
