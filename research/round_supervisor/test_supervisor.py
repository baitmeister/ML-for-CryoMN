"""Regression checks for as-of decisions, measurement integrity and fixed budgets."""

from collections import Counter
import io
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from research.round_supervisor import supervisor as s


@pytest.fixture(scope="module")
def full():
    return s.replay(through=19)


def copy_slate(destination, round_no):
    name = f"ROUND_{round_no:03}"
    proposal = destination / name / "proposal"
    proposal.mkdir(parents=True)
    for filename in ("proposal.csv", "selection_metadata.json", "selection_commit.json"):
        shutil.copyfile(s.ARCHIVE / name / "proposal" / filename, proposal / filename)
    return proposal.parent


def test_prefix_freezes_plan_before_reading_any_results(tmp_path, full):
    for number in range(13, 20):
        directory = copy_slate(tmp_path, number)
        if number < 19:
            (directory / "completed").symlink_to(s.ARCHIVE / directory.name / "completed", target_is_directory=True)
    # An unreadable future payload must have no influence on either the old plans or the next plan.
    (directory / "completed").mkdir()
    (directory / "completed/ingestion_commit.json").write_text("future outcome must stay sealed")
    prefix = s.replay(tmp_path, through=18)
    assert prefix["rounds"] == full["rounds"][:-1]
    assert prefix["pending_plan"] == full["rounds"][-1]["plan"]
    assert prefix["pending_plan"]["requests"]["pair_screened_hit"]["target"][s.VIABILITY] == pytest.approx(64.63)
    assert s.replay(through=12)["rounds"] == []


def test_budgets_caps_and_selective_measurements(full):
    registry_rows = {}
    plans = [r["plan"] for r in full["rounds"]] + [full["pending_plan"]]
    for plan in plans:
        frame, _ = s.frozen_slate(s.ARCHIVE / plan["batch_id"])
        registry_rows.update(frame.set_index("formulation_id").active.to_dict())
        draft = plan["new_project_draft"]
        screen_ids = set(draft["screen_ids"])
        primaries = {x["formulation_id"] for x in draft["compression_priorities"]}
        backups = set(draft["ordered_backup_ids"])
        assert len(screen_ids) == len(draft["screen_ids"]) == 12
        assert len(primaries) == 4 and primaries <= screen_ids
        assert backups <= screen_ids and primaries.isdisjoint(backups)
        assert set(frame[frame.experimental_role.ne("ordinary")].formulation_id) <= screen_ids
        assert max(Counter(k for f in screen_ids for k in registry_rows[f]).values()) <= 5
    assert [sum(r["allocation_evidence"][k] for r in full["rounds"]) for k in ("paired", "failed_gate", "unresolved")] == [18, 4, 6]
    assert sum(x["answered_in"] is not None for x in full["question_ledger"]) == 1
    assert full["totals"]["viability"]["rho"] == pytest.approx(.5420273281)
    assert full["totals"]["force"]["n"] == 30
    assert full["rounds"][-1]["hypervolume"] == pytest.approx(85.43215786)
    assert full["pending_plan"]["requests"]["repeat_frontier"]["frozen_role"] == "off slate"


def test_supplied_bundle_is_idempotent_and_rejects_rewritten_evidence(tmp_path):
    directory = copy_slate(tmp_path, 19)
    source = s.ARCHIVE / directory.name / "completed"
    bundle = tmp_path / "inbox"
    shutil.copytree(source, bundle)
    receipt = bundle / "ingestion_commit.json"
    expected = s.replay(first=19, through=19)["rounds"]
    assert s.replay(tmp_path, first=19, external=(19, receipt))["rounds"] == expected
    packet = json.loads(receipt.read_text())
    rows = pd.read_csv(io.StringIO(packet["batch_observations_csv"]))
    ix = rows.index[rows.endpoint.eq(s.VIABILITY)][0]
    rows.loc[ix, "value"] += 1
    packet["batch_observations_csv"] = rows.to_csv(index=False)
    receipt.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="differ from the completed worksheet or raw traces"):
        s.replay(tmp_path, first=19, external=(19, receipt))
    packet["settings"]["metadata_sha256"] = "0" * 64
    receipt.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="different proposal"):
        s.replay(tmp_path, first=19, external=(19, receipt))


def test_review_can_change_a_slot_but_never_see_future_evidence(full):
    seen = []
    def agent(message):
        seen.append((message["role"], message["packet"]["round"], message["packet"]["evidence_through"]))
        if message["role"] == "critic":
            return {"verdict": "accept", "reason": "The slot choice can be evaluated on the observed frontier."}
        return {"decision": "release_confirmation_slot", "question": "Does a new point extend the frontier?",
                "alternative": "Batch variability might explain a single gain.", "if_supported": "Seek independent confirmation.",
                "if_refuted": "Revisit the best existing point.", "rationale": "Use this discretionary slot for discovery.",
                "reference_ids": ["last_round"]}
    reviewed = s.replay(first=19, through=19, agent_runner=agent, agent_round=20)["pending_plan"]
    assert seen == [("scientist", "ROUND_020", "ROUND_019"), ("critic", "ROUND_020", "ROUND_019")]
    assert reviewed["agent_review"]["status"] == "accepted"
    assert len(reviewed["agent_review"]["packet_sha256"]) == 64
    assert not reviewed["new_project_draft"]["added_screen_ids"]
    assert full["pending_plan"]["new_project_draft"]["added_screen_ids"]


def test_recorded_research_question_reproduces_on_the_archived_cutoff():
    record = json.loads(Path(__file__).with_name("round20_agent_review.json").read_text())
    plan = s.replay(through=19, review_record=record)["pending_plan"]
    review = plan["agent_review"]
    assert review["decision"] == "keep_default" and review["status"] == "accepted"
    assert review["question"] == record["exchanges"][0]["response"]["question"]
    assert review["critique"] == record["exchanges"][1]["response"]["reason"]
