import pytest

from customer_care.api import create_app


def test_api_reports_optional_dependency_requirement_when_fastapi_missing() -> None:
    try:
        import fastapi  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="uv sync --extra web"):
            create_app()


def test_demo_endpoints_execute_scripted_graph_scenarios() -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    client = TestClient(create_app())

    health = client.get("/health")
    workbench = client.get("/")
    scenarios = client.get("/v1/demo/scenarios")
    multi = client.post("/v1/demo/multi-skill")
    conflict = client.post("/v1/demo/conflict")
    side_effect = client.post("/v1/demo/side-effect")
    multi_confirmation = client.post("/v1/demo/multi-confirmation")

    assert health.status_code == 200
    assert health.json()["service"] == "customer-care"
    assert workbench.status_code == 200
    assert "客服 Agent / 学习工作台" in workbench.text
    assert scenarios.json()["mode"] == "scripted_mock"
    assert multi.json()["trace"]["execution_status"] == "succeeded"
    assert multi.json()["demo"]["mode"] == "scripted_mock"
    assert conflict.json()["trace"]["route"] == "clarify"
    pending = side_effect.json()
    assert pending["trace"]["route"] == "awaiting_confirmation"
    token = pending["trace"]["confirmation"]["token"]
    confirmed = client.post("/v1/demo/side-effect/confirm", json={"token": token})
    replayed = client.post("/v1/demo/side-effect/confirm", json={"token": token})
    assert confirmed.status_code == 200
    assert confirmed.json()["trace"]["confirmation_status"] == "completed"
    assert confirmed.json()["trace"]["execution_status"] == "succeeded"
    assert replayed.status_code == 409
    pending_multi = multi_confirmation.json()
    cancelled = client.post(
        "/v1/demo/multi-confirmation/cancel",
        json={"token": pending_multi["trace"]["confirmation"]["token"]},
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["trace"]["confirmation_status"] == "declined"
    assert cancelled.json()["trace"]["execution_status"] == "cancelled"
    assert client.post("/v1/demo/not-a-scenario").status_code == 404
