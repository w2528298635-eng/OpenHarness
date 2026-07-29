from openharness.repopilot.events import RunEvent, RunEventKind, redact_and_bound


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


def test_redaction_sanitizes_nested_values_without_mutating_input() -> None:
    original = {
        "headers": {"Authorization": "Bearer abcdefghijklmnop"},
        "items": ["token=very-secret-token", {"password": "hunter2"}],
    }

    sanitized = redact_and_bound(original)

    assert original["headers"]["Authorization"] == "Bearer abcdefghijklmnop"
    assert "abcdefghijklmnop" not in str(sanitized)
    assert "very-secret-token" not in str(sanitized)
    assert "hunter2" not in str(sanitized)
    assert str(sanitized).count("[REDACTED]") == 3
