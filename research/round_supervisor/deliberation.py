"""Optional, evidence-bounded scientist and critic review of one round decision."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


PROSE = ("question", "alternative", "if_supported", "if_refuted", "rationale")
TIMEOUT_SECONDS = 60
MAX_PROSE = 800
MAX_REPLY_BYTES = 16_384

SCIENCE_BRIEF = (
    "Act as the round scientist for the stated research goal. Use only the supplied evidence packet. "
    "Choose one eligible decision. Ask a falsifiable question, identify a competing explanation, and "
    "describe what future evidence would support or refute the question and how each result would "
    "change the next decision. Distinguish measured from forecast values and cite applicable packet "
    "reference_ids. Keep each answer under 800 characters. Return one JSON object matching output_schema."
)
CRITIC_BRIEF = (
    "Independently assess the proposed card against the original evidence packet and research goal. "
    "Check its references, measurement units, missing evidence, alternative explanation, and whether "
    "the proposed observation could change the decision. Accept a supported, testable card or request "
    "one specific revision. Use only the packet. Reply with one JSON object matching output_schema."
)


class ReviewError(ValueError):
    """A diagnostic safe to include in a round record."""


def packet_digest(packet):
    return hashlib.sha256(json.dumps(packet, sort_keys=True, allow_nan=False).encode()).hexdigest()


def recorded_review(packet, record):
    """Replay the two reviewers' recorded answers only against their original evidence cutoff."""
    if record["round"] != packet["round"] or record["packet_sha256"] != packet_digest(packet):
        raise ValueError("The recorded review does not match this round and evidence packet")
    turns = iter(record["exchanges"])

    def respond(message):
        turn = next(turns)
        if turn["role"] != message["role"]:
            raise ReviewError("Recorded agent roles do not match the review sequence.")
        return turn["response"]

    review = deliberate(packet, respond)
    if next(turns, None) is not None or review["status"] != record["status"]:
        raise ValueError("The recorded exchanges did not reproduce their recorded outcome")
    return review


def _schema(fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def _runner_reply(runner, message):
    if callable(runner):
        return runner(copy.deepcopy(message))

    program = os.fspath(runner)
    executable = shutil.which(program)
    if executable is None:
        raise ReviewError("The configured agent executable could not be found.")
    executable = str(Path(executable).resolve())
    with tempfile.TemporaryDirectory(prefix="cryomn-review-") as directory:
        work = Path(directory)
        output_path = work / "answer.json"
        args = [executable]
        if program == "codex":
            schema_path = work / "schema.json"
            schema_path.write_text(json.dumps(message["output_schema"]), encoding="utf-8")
            args += ["--no-daemon", "--ask-for-approval", "never", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                     "--sandbox", "read-only", "--skip-git-repo-check", "--cd", directory,
                     "--disable", "shell_tool", "--disable", "unified_exec", "--disable", "apps",
                     "--disable", "browser_use", "--disable", "computer_use", "--disable", "code_mode_host",
                     "--disable", "memories", "--disable", "plugins", "--disable", "multi_agent",
                     "-c", 'web_search="disabled"', "--output-schema", str(schema_path),
                     "--output-last-message", str(output_path), "-"]
        with (work / "stdout").open("w+b") as stdout:
            subprocess.run(args, input=json.dumps(message, allow_nan=False).encode("utf-8"), cwd=directory,
                           stdout=stdout, stderr=subprocess.DEVNULL, timeout=TIMEOUT_SECONDS, check=True)
            if program == "codex":
                with output_path.open("rb") as output:
                    body = output.read(MAX_REPLY_BYTES + 1)
            else:
                stdout.seek(0)
                body = stdout.read(MAX_REPLY_BYTES + 1)
        if len(body) > MAX_REPLY_BYTES:
            raise ReviewError("The agent reply exceeds the response size limit.")
        try:
            return json.loads(body)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ReviewError("The agent returned invalid JSON.") from exc


def _prose(value):
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_PROSE:
        raise ReviewError("A required agent explanation was missing or exceeded the length limit.")
    text = " ".join(value.split())
    if not text.isprintable():
        raise ReviewError("An agent explanation contained invalid characters.")
    return text


def _object(reply, fields):
    if not isinstance(reply, dict) or reply.keys() != fields.keys():
        raise ReviewError("The agent reply did not match the required fields.")
    return reply


def deliberate(packet, runner):
    """Use a callable, trusted JSON executable, or opt-in 'codex'. Fall back to the eligible default."""
    choices, refs, default = packet["eligible_decisions"], packet["reference_ids"], packet["default_decision"]
    if not isinstance(default, str) or default not in choices:
        raise ValueError("The packet must specify an eligible default decision")
    attempts, reviews = 0, []

    def fallback(reason):
        return {"status": "fallback", "decision": default, **dict.fromkeys(PROSE), "rationale": reason,
                "reference_ids": [], "critique": reviews[-1]["reason"] if reviews else None,
                "attempts": attempts, "reviews": reviews}

    if not refs:
        return fallback("No evidence references were available for this decision.")
    text = {"type": "string"}
    science_fields = {"decision": {"type": "string", "enum": list(choices)}, **dict.fromkeys(PROSE, text),
                      "reference_ids": {"type": "array", "items": {"type": "string", "enum": list(refs)}}}
    critic_fields = {"verdict": {"type": "string", "enum": ["accept", "revise"]}, "reason": text}
    scientist = {"role": "scientist", "instructions": SCIENCE_BRIEF, "packet": packet,
                 "output_schema": _schema(science_fields)}
    try:
        for attempts in (1, 2):
            raw = _object(_runner_reply(runner, scientist), science_fields)
            if raw["decision"] not in choices:
                raise ReviewError("The scientist selected an ineligible decision.")
            cited = raw["reference_ids"]
            if not isinstance(cited, list) or not cited or any(not isinstance(i, str) or i not in refs for i in cited):
                raise ReviewError("The scientist must cite evidence present in the packet.")
            card = {"decision": raw["decision"], "reference_ids": list(dict.fromkeys(cited)),
                    **{field: _prose(raw[field]) for field in PROSE}}
            critic = {"role": "critic", "instructions": CRITIC_BRIEF, "packet": packet, "proposal": card,
                      "output_schema": _schema(critic_fields)}
            review = _object(_runner_reply(runner, critic), critic_fields)
            if review["verdict"] not in ("accept", "revise"):
                raise ReviewError("The critic returned an unrecognized verdict.")
            reason = _prose(review["reason"])
            reviews.append({"verdict": review["verdict"], "reason": reason})
            if review["verdict"] == "accept":
                return {**card, "status": "accepted", "critique": reason, "attempts": attempts, "reviews": reviews}
            scientist["revision"] = {"previous_proposal": card, "critic_reason": reason}
        return fallback("The critic requested further revision after the allowed revision cycle.")
    except ReviewError as exc:
        return fallback(str(exc))
    except subprocess.TimeoutExpired:
        return fallback("The agent exceeded its time limit.")
    except Exception:
        # Runner errors may contain raw evidence or credentials. Only return a stable diagnostic.
        return fallback("The agent runner could not complete the review.")
