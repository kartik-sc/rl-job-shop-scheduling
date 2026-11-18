import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api.main import app

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "shared" / "schemas" / "event_log.schema.json"


def test_health() -> None:
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_mock_episode_matches_schema_and_plan_constraints() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    allowed_regimes = set(schema["properties"]["meta"]["properties"]["regime"]["enum"])
    allowed_policies = set(schema["properties"]["meta"]["properties"]["policy"]["enum"])
    allowed_event_types = set(
        schema["properties"]["events"]["items"]["properties"]["type"]["enum"]
    )
    required_event_keys = set(schema["properties"]["events"]["items"]["required"])

    client = TestClient(app)
    res = client.get("/episodes/mock")
    assert res.status_code == 200
    payload = res.json()

    assert set(schema["required"]).issubset(payload.keys())
    meta = payload["meta"]
    assert set(schema["properties"]["meta"]["required"]).issubset(meta.keys())
    assert meta["regime"] in allowed_regimes
    assert meta["policy"] in allowed_policies
    assert meta["n_machines"] == 6

    events = payload["events"]
    assert len(events) > 0

    # Chronological order
    times = [e["t"] for e in events]
    assert times == sorted(times)

    for ev in events:
        assert required_event_keys.issubset(ev.keys())
        assert ev["type"] in allowed_event_types
        if "machine" in ev:
            assert 0 <= ev["machine"] < meta["n_machines"]
        if "risk" in ev:
            assert 0.0 <= ev["risk"] <= 1.0

    op_ends = [e for e in events if e["type"] == "op_end"]
    maint_starts = [e for e in events if e["type"] == "maintenance_start"]
    maint_ends = [e for e in events if e["type"] == "maintenance_end"]
    breakdowns = [e for e in events if e["type"] == "breakdown"]
    ep_ends = [e for e in events if e["type"] == "episode_end"]

    # Day 1-2 Plan requirements: 6 machines, 20 operations, 1 maintenance, 1 breakdown
    assert len(op_ends) == 20
    assert len(maint_starts) == 1
    assert len(maint_ends) == 1
    assert len(breakdowns) == 1
    assert len(ep_ends) == 1


def test_episode_list_puts_real_logs_first_and_labels_the_mock() -> None:
    episodes = TestClient(app).get("/episodes").json()
    real = [e for e in episodes if e["source"] == "real"]
    mock = [e for e in episodes if e["source"] == "mock"]

    assert {e["meta"]["policy"] for e in real} == {"spt", "lpt", "fifo", "threshold_spt"}
    assert episodes[: len(real)] == real
    assert all(e["label"].endswith(", real health model") for e in real)
    assert "Threshold+SPT, real health model" in {e["label"] for e in real}
    assert all(not e["label"].startswith("MOCK DATA") for e in real)
    assert [e["id"] for e in mock] == ["mock"]
    assert all(e["label"].startswith("MOCK DATA") for e in mock)


def test_real_episodes_are_served_and_fit_the_schema() -> None:
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    client = TestClient(app)
    for summary in client.get("/episodes").json():
        if summary["source"] != "real":
            continue
        log = client.get(f"/episodes/{summary['id']}").json()
        jsonschema.validate(log, schema)
        assert log["meta"]["health_model"] == "cmapss/ocsvm/A_ds_iso"
        assert log["meta"]["n_machines"] == 15
        assert log["events"][-1]["type"] == "episode_end"
        assert summary["makespan"] == log["events"][-1]["t"]


def test_unknown_or_escaping_episode_ids_are_not_found() -> None:
    client = TestClient(app)
    assert client.get("/episodes/nope").status_code == 404
    assert client.get("/episodes/..%2Fsummary").status_code == 404
