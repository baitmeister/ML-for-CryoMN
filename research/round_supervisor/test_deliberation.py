"""Check the agent handoff without contacting a model or a lab."""

import json
import subprocess
from pathlib import Path

import pytest

from research.round_supervisor.deliberation import MAX_PROSE, deliberate, packet_digest, recorded_review


PACKET = {
    "goal": "Evaluate the two measured endpoints",
    "round": "ROUND_019", "evidence_through": "ROUND_018",
    "eligible_decisions": ["keep_default", "release_confirmation_slot"],
    "default_decision": "keep_default", "reference_ids": ["r18_gate", "r18_force"],
    "evidence": {"r18_gate": "One primary failed formation", "r18_force": "Four force results"},
}
CARD = {
    "decision": "release_confirmation_slot", "reference_ids": ["r18_gate"],
    "question": "Will every primary pass formation this round?", "alternative": "Formation could vary by batch.",
    "if_supported": "Consider a repeat next round.", "if_refuted": "Retain the gate follow-up.",
    "rationale": "A prior primary failed the gate.",
}
ACCEPT = {"verdict": "accept", "reason": "The proposed test uses a recorded gate outcome."}
REVISE = {"verdict": "revise", "reason": "Specify how a missing formation result is handled."}


def sequence(*responses):
    calls = []

    def run(message):
        calls.append(message)
        return responses[len(calls) - 1]

    return run, calls


def test_independent_critic_and_single_revision():
    revised = {**CARD, "if_refuted": "Retain the gate follow-up and defer missing results."}
    run, calls = sequence(CARD, REVISE, revised, ACCEPT)
    result = deliberate(PACKET, run)
    assert [m["role"] for m in calls] == ["scientist", "critic", "scientist", "critic"]
    assert all(m["packet"] == PACKET for m in calls)
    assert "revision" not in calls[0] and "revision" not in calls[3]
    assert calls[2]["revision"]["critic_reason"] == REVISE["reason"]
    assert calls[3]["proposal"] == revised
    assert result["status"] == "accepted" and result["decision"] == revised["decision"]
    assert result["reviews"] == [REVISE, ACCEPT] and result["attempts"] == 2


@pytest.mark.parametrize("changes", [
    {"decision": "invent_new_trial"}, {"reference_ids": []}, {"reference_ids": ["future"]},
    {"rationale": "x" * (MAX_PROSE + 1)}, {"alternative": ""}, {"if_supported": "ok\x00"},
    {"unexpected": "unstructured output"},
])
def test_out_of_scope_or_malformed_card_uses_default(changes):
    run, calls = sequence({**CARD, **changes})
    result = deliberate(PACKET, run)
    assert len(calls) == 1
    assert result["status"] == "fallback" and result["decision"] == PACKET["default_decision"]
    assert result["question"] is None


def test_critic_rejection_and_runner_errors():
    run, calls = sequence(CARD, REVISE, CARD, REVISE)
    assert deliberate(PACKET, run)["status"] == "fallback" and len(calls) == 4

    def broken(_):
        raise RuntimeError("private stderr, raw data, or credentials")

    result = deliberate(PACKET, broken)
    assert "private" not in json.dumps(result)
    assert result["decision"] == PACKET["default_decision"]
    assert deliberate({**PACKET, "reference_ids": []}, broken)["attempts"] == 0


def test_record_replays_on_its_original_evidence_only():
    record = {"round": PACKET["round"], "packet_sha256": packet_digest(PACKET), "status": "accepted",
              "exchanges": [{"role": "scientist", "response": CARD}, {"role": "critic", "response": ACCEPT}]}
    assert recorded_review(PACKET, record)["decision"] == CARD["decision"]
    with pytest.raises(ValueError, match="does not match"):
        recorded_review({**PACKET, "evidence_through": "ROUND_019"}, record)


def test_timeout_and_malformed_critic_use_default():
    def delayed(_):
        raise subprocess.TimeoutExpired("private executable", 60)

    result = deliberate(PACKET, delayed)
    assert result["decision"] == PACKET["default_decision"] and "time limit" in result["rationale"]
    assert "private" not in json.dumps(result)
    run, calls = sequence(CARD, {"verdict": "skip", "reason": "I did not review it."})
    assert deliberate(PACKET, run)["status"] == "fallback" and len(calls) == 2


def test_trusted_local_executable(tmp_path):
    program = tmp_path / "agent with space"
    program.write_text("#!/usr/bin/env python3\nimport json, sys\nrequest = json.load(sys.stdin)\n"
                       f"print(json.dumps({CARD!r} if request['role'] == 'scientist' else {ACCEPT!r}))\n")
    program.chmod(0o700)
    result = deliberate(PACKET, program)
    assert result["status"] == "accepted" and result["attempts"] == 1


def test_codex_invocation_is_isolated_and_ephemeral(monkeypatch):
    from research.round_supervisor import deliberation

    monkeypatch.setattr(deliberation.shutil, "which", lambda name: "/mock/codex")
    directories = []

    def fake_run(argv, **options):
        cwd = Path(options["cwd"])
        directories.append(cwd)
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert argv[argv.index("--cd") + 1] == str(cwd)
        assert "--ephemeral" in argv and "--ignore-user-config" in argv and "--ignore-rules" in argv
        assert {"memories", "plugins", "shell_tool", "apps"}.issubset(argv)
        assert options["timeout"] == deliberation.TIMEOUT_SECONDS and options["stderr"] is subprocess.DEVNULL
        msg = json.loads(options["input"])
        schema = json.loads(Path(argv[argv.index("--output-schema") + 1]).read_text())
        assert schema == msg["output_schema"]
        answer = CARD if msg["role"] == "scientist" else ACCEPT
        Path(argv[argv.index("--output-last-message") + 1]).write_text(json.dumps(answer))

    monkeypatch.setattr(deliberation.subprocess, "run", fake_run)
    assert deliberate(PACKET, "codex")["status"] == "accepted"
    assert len(set(directories)) == 2 and all(not cwd.exists() for cwd in directories)
