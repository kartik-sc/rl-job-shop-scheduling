"""Event logs from the real health model fit the committed schema."""

import json

import jsonschema

from backend.evaluation.actors import reference_rule, threshold
from backend.evaluation.demo import SCHEMA_PATH, Row, run, schema_events, write_log
from backend.training.protocol import REPO_ROOT, load_protocol

PROTOCOL = load_protocol(REPO_ROOT / "configs" / "protocol.toml")


def test_repair_end_and_offline_risk_are_dropped() -> None:
    events = [
        {"t": 0, "type": "risk_update", "machine": 0, "risk": 0.7},
        {"t": 0, "type": "op_start", "machine": 0, "job": 0, "op": 0},
        {"t": 5, "type": "breakdown", "machine": 0, "job": 0, "op": 0},
        {"t": 6, "type": "risk_update", "machine": 0, "risk": 0.9},
        {"t": 6, "type": "risk_update", "machine": 1, "risk": 0.6},
        {"t": 9, "type": "repair_end", "machine": 0},
        {"t": 9, "type": "risk_update", "machine": 0, "risk": 0.65},
    ]
    kept = schema_events(events)
    assert [e["type"] for e in kept] == ["risk_update", "op_start", "breakdown", "risk_update", "risk_update"]
    assert [e["t"] for e in kept if e.get("machine") == 0 and e["type"] == "risk_update"] == [0, 9]


def test_logs_from_every_policy_validate_and_carry_no_rul(tmp_path) -> None:  # type: ignore[no-untyped-def]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for contender in (reference_rule("spt"), threshold(0.8)):
        row, events = run(PROTOCOL, contender, "high", "ta06", 2000, "test")
        assert isinstance(row, Row)
        path = tmp_path / f"{row.policy}.json"
        write_log(path, row, events, schema)
        document = json.loads(path.read_text())
        jsonschema.validate(document, schema)
        assert all("rul" not in key for event in document["events"] for key in event)
        assert all(0.0 <= e["risk"] <= 1.0 for e in document["events"] if "risk" in e)
