"""Replay a CryoMN supervisor against frozen proposals and round-local validation receipts."""

import argparse
from collections import Counter
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/08_multi_objective"))
from helper.artifacts import sha256_file, validate_completed_against_proposal
from helper.endpoints import INTACT_PATCH_ENDPOINT as GATE, aggregate_intact_patch_replicates
from helper.evaluation_metrics import compute_fixed_reference_hypervolume, compute_observed_pareto_front
from helper.feedback import ingest_feedback
from helper.registry import load_registry
from helper.selection_constraints import _active_ingredient_set
from helper.similarity import is_retest_row
from helper.terminal_force import DEFINITION, MODEL_FIELD as FORCE

VIABILITY = "viability_percent"
UNITS = {VIABILITY: "percent", GATE: "binary", FORCE: "N_per_needle"}
ARCHIVE = ROOT / "results/multi_objective_v2/rounds"
POLICY = "msc_joint_round_review_v1"
REFERENCE = {VIABILITY: 0.0, FORCE: 0.0}
REGISTRY = load_registry()


def records(frame):
    """Return native JSON numbers and null for a missing measurement."""
    return json.loads(frame.to_json(orient="records", double_precision=14))


def frozen_slate(directory):
    manifest = json.loads((directory / "proposal/selection_commit.json").read_text())["hashes"]
    contents = {}
    for name in ("proposal/proposal.csv", "proposal/selection_metadata.json"):
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest[name]:
            raise ValueError(f"{directory.name} frozen file differs from its selection receipt: {name}")
        contents[name] = data
    metadata = json.loads(contents["proposal/selection_metadata.json"])
    if metadata["active_phase"] != "mechanics_enabled" or metadata["batch_id"] != directory.name:
        raise ValueError(f"{directory.name} requires a matching full-mechanics proposal")

    columns = ["batch_id", "formulation_id", "candidate_id", "selection_rank", "experimental_role",
               "mechanical_backup_status", "mechanical_selection_rank", "predicted_viability_percent", "viability_std",
               "predicted_critical_axial_load_N_per_needle", "critical_axial_load_std",
               "mechanical_prediction_definition_id", "mechanical_prediction_status", "recommendation_type",
               "candidate_origin", "support_status"]
    # Only proposal-time fields enter the decision, even if the worksheet format gains columns.
    frame = pd.read_csv(io.BytesIO(contents["proposal/proposal.csv"]),
                        usecols=columns + REGISTRY.feature_names)
    ids = ["candidate_id", "formulation_id", "selection_rank", "experimental_role"]
    if (frame.empty or not frame.batch_id.eq(directory.name).all() or frame[ids].isna().any().any()
            or any(frame[key].duplicated().any() for key in ids[:3])):
        raise ValueError(f"{directory.name} has ambiguous proposal identities")
    if not frame.mechanical_prediction_definition_id.eq(DEFINITION).all():
        raise ValueError(f"{directory.name} uses a different mechanical forecast definition")
    frame = frame.rename(columns={"predicted_viability_percent": "viability_forecast",
                                  "predicted_critical_axial_load_N_per_needle": "force_forecast"})
    frame.loc[frame.mechanical_prediction_status.eq("untrained_placeholder"), "force_forecast"] = np.nan
    frame["control"] = frame.experimental_role.eq("campaign_control")
    frame["primary"] = frame.mechanical_backup_status.eq("primary")
    frame["mechanical_role"] = frame.mechanical_backup_status.fillna("screen only")
    ingredients = [i for i in REGISTRY.ingredients if i.feature_name in frame]

    def composition(row):
        parts = []
        for ingredient in ingredients:
            value = row[ingredient.feature_name]
            if pd.notna(value) and value > 0:
                unit = "%" if ingredient.unit == "pct" else " M"
                parts.append(f"{value:.5g}{unit} {ingredient.display_name}")
        return ", ".join(parts) or "no active registry ingredients"

    frame["recipe"] = frame.apply(composition, axis=1)
    frame["active"] = [sorted(_active_ingredient_set(row, REGISTRY)) for _, row in frame.iterrows()]
    frame["repeat_role"] = [is_retest_row(row) for _, row in frame.iterrows()]
    frame.attrs["caps"] = {"ingredient": metadata["ingredient_frequency_diversity"]["max_rows_per_ingredient"],
                          "shared_pair": metadata["shared_ingredient_pair_diversity"]["max_non_retest_rows_per_pair"],
                          "boundary": metadata["boundary_slate_cap"]["max_support_boundary_rows"]}
    return frame.drop(columns=REGISTRY.feature_names), manifest["proposal/proposal.csv"]


def open_results(path, directory, proposal, digest):
    data = path.read_bytes()
    receipt = json.loads(data)
    settings = receipt["settings"]
    metadata_file = directory / "proposal/selection_metadata.json"
    if (receipt["version"] != 1 or settings["batch_id"] != directory.name or settings["proposal_sha256"] != digest
            or settings["metadata_sha256"] != sha256_file(metadata_file)):
        raise ValueError(f"{directory.name} validation receipt belongs to a different proposal")
    bundle = path.parent
    hashes = {Path(name).name: value for name, value in receipt["frozen_outputs"].items()}
    for name in ("completed.csv", "finalization.json", "mechanical_execution_manifest.json"):
        if sha256_file(bundle / name) != hashes[name]:
            raise ValueError(f"{directory.name} changed completed artifact: {name}")
    worksheet = bundle / "completed.csv"
    if hashes["completed.csv"] != receipt["worksheet_sha256"]:
        raise ValueError(f"{directory.name} completed worksheet differs from the validation receipt")
    finalization = json.loads((bundle / "finalization.json").read_text())
    if finalization["completion_attested"] is not True or (finalization["unmeasured_outcomes"] and not finalization["unmeasured_reason"].strip()):
        raise ValueError(f"{directory.name} requires attested completion and an explanation for outstanding measures")
    proposal_path = directory / "proposal/proposal.csv"
    validate_completed_against_proposal(worksheet, proposal_path)
    observations = pd.read_csv(io.StringIO(receipt["batch_observations_csv"]))
    keys = ["formulation_id", "endpoint", "replicate_id"]
    if (observations.empty or not observations.batch_id.eq(directory.name).all()
            or observations.observation_id.isna().any() or observations.observation_id.duplicated().any()
            or observations[keys].isna().any().any() or observations.duplicated(keys).any()
            or not observations.formulation_id.isin(proposal.formulation_id).all()):
        raise ValueError(f"{directory.name} has missing, duplicated, or unmatched observation identities")
    data_rows = observations.loc[observations.endpoint.isin(UNITS)].copy()
    data_rows["value"] = pd.to_numeric(data_rows.value, errors="raise")
    if (data_rows.empty or not np.isfinite(data_rows.value).all()
            or not data_rows.unit.eq(data_rows.endpoint.map(UNITS)).all()):
        raise ValueError(f"{directory.name} has an invalid value or unit")
    # Parse the attested worksheet and its raw traces in memory, independently of the embedded extract.
    _, parsed = ingest_feedback(worksheet, [proposal_path], pd.DataFrame(), pd.DataFrame(), REGISTRY,
                                directory.name, batch_date=settings.get("batch_date", ""),
                                default_needles_compressed=settings.get("default_needles_compressed"),
                                viability_noise=settings["viability_noise"], proposal_metadata=json.loads(metadata_file.read_text()),
                                round_mechanical_endpoint=settings.get("mechanical_endpoint"))
    fields = ["observation_id", *keys, "value", "unit", "mechanical_definition_id", "raw_file_hash",
              "preparation_id", "specimen_id", "cell_batch_id"]
    def comparable(rows):
        table = rows.loc[rows.endpoint.isin(UNITS), fields].copy()
        table["value"] = pd.to_numeric(table.value)
        return table.fillna("").set_index("observation_id").sort_index()
    try:
        pd.testing.assert_frame_equal(comparable(data_rows), comparable(parsed), check_dtype=False, rtol=1e-12, atol=1e-12)
    except AssertionError as error:
        raise ValueError(f"{directory.name} observations differ from the completed worksheet or raw traces") from error
    data_rows = data_rows.sort_values(["formulation_id", "endpoint", "replicate_id"])
    frame = proposal.copy()
    assay_rows = data_rows[data_rows.endpoint.isin([VIABILITY, FORCE])]
    for identity in ("preparation_id", "cell_batch_id"):
        recorded = assay_rows.groupby("formulation_id")[identity].nunique()
        frame[identity + "_count"] = frame.formulation_id.map(recorded).fillna(0).astype(int)
    for endpoint in UNITS:
        rows = data_rows[data_rows.endpoint.eq(endpoint)]
        if endpoint == FORCE and (not rows.mechanical_definition_id.eq(DEFINITION).all() or rows.value.lt(0).any()):
            raise ValueError(f"{directory.name} has an incompatible terminal-force observation")
        if endpoint == VIABILITY and not rows.value.between(0, 100).all():
            raise ValueError(f"{directory.name} viability lies outside 0 to 100 percent")
        if endpoint == GATE and not rows.value.isin([0, 1]).all():
            raise ValueError(f"{directory.name} formation result must be 0 or 1")
        grouped = rows.groupby("formulation_id").value
        summary = grouped.agg(aggregate_intact_patch_replicates) if endpoint == GATE else grouped.mean()
        frame[endpoint] = frame.formulation_id.map(summary)
        if endpoint != GATE:
            frame[endpoint + "_readouts"] = frame.formulation_id.map(grouped.agg(list))
    if (frame[FORCE].notna() & ~frame[GATE].eq(1)).any():
        raise ValueError(f"{directory.name} measured force requires a same-batch all-pass gate")
    return frame, hashlib.sha256(data).hexdigest()


def pareto_evidence(history):
    paired = history.loc[~history.control & history[GATE].eq(1)].dropna(subset=[VIABILITY, FORCE])
    ranked = compute_observed_pareto_front(paired.reset_index(drop=True))
    area = compute_fixed_reference_hypervolume(ranked, REFERENCE)["hypervolume"]
    front = ranked[ranked.is_pareto].sort_values([VIABILITY, FORCE], ascending=False)
    return front, len(paired), None if pd.isna(area) else float(area)


POINT = ["batch_id", "formulation_id", "selection_rank", "recipe", "active", "support_status", VIABILITY, FORCE,
         "viability_forecast", "force_forecast", VIABILITY + "_readouts", FORCE + "_readouts",
         "preparation_id_count", "cell_batch_id_count"]


def next_agenda(latest, history, front):
    screened = latest.loc[~latest.control & latest[GATE].eq(1) & latest[FORCE].isna()].dropna(subset=[VIABILITY])
    screened = screened.sort_values([VIABILITY, "selection_rank"], ascending=[False, True])
    paired = history.loc[~history.control & history[GATE].eq(1)].dropna(subset=[VIABILITY, FORCE])
    batches = paired.groupby("formulation_id").batch_id.nunique()
    single_batch = front.loc[front.formulation_id.map(batches).eq(1)]
    priorities = {}
    for kind, candidates in [("pair_screened_hit", screened), ("repeat_frontier", single_batch)]:
        if not candidates.empty:
            priorities[kind] = records(candidates.head(1)[POINT])[0]
    primary = latest.loc[latest.primary, GATE]
    if primary.isna().any():
        focus = "resolve missing primary formation evidence"
    elif primary.eq(0).any():
        focus = "check whether all primary patches form intact and preserve ordered backups"
    else:
        focus = "seek an independent batch for the leading measured trade-off" if len(single_batch) else "extend the measured trade-off"
    return {"evidence_through": latest.batch_id.iloc[0], "focus": focus, "requests": priorities,
            "reserve_confirmation": bool(len(primary) and primary.eq(1).all() and len(single_batch))}


def draft_allocation(slate, agenda):
    """Give experimental questions a budget, filling other slots from the frozen optimizer ordering."""
    primaries = slate.loc[slate.primary].sort_values("mechanical_selection_rank")
    requested = [(kind, target) for kind, target in agenda["requests"].items()
                 if kind != "repeat_frontier" or agenda["reserve_confirmation"]]
    # Try both questions together before deferring one. High-priority questions win ties.
    for size in range(min(len(requested), len(primaries)), -1, -1):
        for selected in combinations(requested, size):
            allocation = fit_screen_budget(slate, selected)
            if allocation is not None:
                break
        else:
            continue
        break
    else:
        raise ValueError("Even the frozen slate violates the screening caps")
    screens, added, dropped = allocation
    mechanical = {}
    for kind, target in selected:
        key = target["formulation_id"]
        if key in mechanical:
            mechanical[key]["purpose"] += " + " + kind
        else:
            mechanical[key] = {"formulation_id": key, "recipe": target["recipe"], "purpose": kind}
    ordered = slate.loc[slate.mechanical_selection_rank.notna() & slate.formulation_id.isin(screens)].sort_values("mechanical_selection_rank")
    for row in ordered.itertuples():
        if len(mechanical) == len(primaries):
            break
        mechanical.setdefault(row.formulation_id, {"formulation_id": row.formulation_id, "recipe": row.recipe,
                                                   "purpose": "optimizer primary"})
    if len(mechanical) != len(primaries):
        raise ValueError("Too few eligible mechanical candidates for the proposed capacity")
    return {"screen_ids": screens, "compression_priorities": list(mechanical.values()),
            "ordered_backup_ids": [key for key in ordered.formulation_id if key not in mechanical],
            "deferred_by_constraints": [kind for kind, _ in requested if kind not in dict(selected)],
            "added_screen_ids": added, "displaced_screen_ids": dropped,
            "displaced_compression": records(primaries.loc[~primaries.formulation_id.isin(mechanical),
                                                          ["formulation_id", "selection_rank", "recipe", "viability_forecast", "force_forecast"]])}


def fit_screen_budget(slate, selected):
    caps = slate.attrs["caps"]
    originals = slate.sort_values("selection_rank").to_dict("records")
    original_ids = set(slate.formulation_id)
    targets = {point["formulation_id"]: point for _, point in selected}
    additions = [dict(p, repeat_role=True) for k, p in targets.items() if k not in original_ids]
    removable = slate.loc[slate.experimental_role.eq("ordinary") & ~slate.formulation_id.isin(targets)]
    removable = removable.sort_values(["primary", "selection_rank"], ascending=[True, False])
    for dropped in combinations(removable.formulation_id, len(additions)):
        rows = [r for r in originals if r["formulation_id"] not in dropped] + additions
        frequency = Counter(k for r in rows for k in r["active"])
        ordinary_sets = [tuple(r["active"]) for r in rows if not r["repeat_role"] and len(r["active"]) >= 2]
        shared = Counter(pair for combo in ordinary_sets for pair in combinations(combo, 2))
        if (max(frequency.values(), default=0) > caps["ingredient"] or max(shared.values(), default=0) > caps["shared_pair"]
                or sum(r["support_status"] == "boundary" for r in rows) > caps["boundary"]
                or any(n > (2 if len(combo) == 2 else 1) for combo, n in Counter(ordinary_sets).items())):
            continue
        return [r["formulation_id"] for r in rows], [r["formulation_id"] for r in additions], list(dropped)
    return None


def freeze_plan(proposal, digest, agenda):
    ranked = proposal.loc[~proposal.control].dropna(subset=["viability_forecast"]).sort_values(
        ["viability_forecast", "selection_rank"], ascending=[False, True])
    requests = {}
    for kind, target in agenda["requests"].items():
        match = proposal.loc[proposal.formulation_id.eq(target["formulation_id"])]
        role = match.mechanical_role.iloc[0] if len(match) else "off slate"
        forecast = records(match[["selection_rank", "viability_forecast", "force_forecast"]])
        if kind == "pair_screened_hit":
            question = "Does this intact screened hit extend the prior measured frontier when both endpoints are measured together?"
            consequence = "If a prior point matches or exceeds both means, retain the prior frontier. Otherwise add the new trade-off."
        else:
            question = "How much do both endpoints change for this exact frontier formulation in a separately prepared batch?"
            consequence = "Record both shifts and preparation and batch identities. Apply a reproducibility margin only if the project has fixed one."
        requests[kind] = {"target": target, "question": question, "decision_rule": consequence,
                          "frozen_role": role, "matched_forecast": forecast[0] if forecast else None}
    return {"batch_id": proposal.batch_id.iloc[0], "proposal_sha256": digest, "policy": POLICY,
            "evidence_through": agenda["evidence_through"], "focus": agenda["focus"],
            "requests": requests, "rankable_ids": ranked.formulation_id.tolist(),
            "shortlist": ranked.formulation_id.head(3).tolist(),
            "primary_ids": proposal.loc[proposal.primary, "formulation_id"].tolist(),
            "screens": len(proposal), "new_project_draft": draft_allocation(proposal, agenda)}


def allocation_evidence(plan, observed):
    slots = plan["new_project_draft"]["compression_priorities"]
    outcomes = observed.set_index("formulation_id").reindex([s["formulation_id"] for s in slots])
    # Only historical measurements on the same formulation in the same round provide an outcome.
    measured = outcomes[GATE].eq(1) & outcomes[VIABILITY].notna() & outcomes[FORCE].notna()
    failed = outcomes[GATE].eq(0)
    return {"slots": len(slots), "paired": int(measured.sum()), "failed_gate": int(failed.sum()),
            "unresolved": int((~measured & ~failed).sum())}


def resolve_requests(plan, observed, prior_front):
    resolutions = {}
    for kind, request in plan["requests"].items():
        target = request["target"]
        match = observed.loc[observed.formulation_id.eq(target["formulation_id"])]
        result = {"status": "off slate", "observed": None}
        if len(match):
            row = match.iloc[0]
            result["observed"] = records(match[POINT])[0]
            if pd.isna(row[GATE]):
                result["status"] = "formation missing"
            elif row[GATE] == 0:
                result["status"] = "failed formation"
            elif pd.isna(row[VIABILITY]) or pd.isna(row[FORCE]):
                result["status"] = "paired outcome missing"
            elif kind == "pair_screened_hit":
                already_covered = any(p[VIABILITY] >= row[VIABILITY] and p[FORCE] >= row[FORCE] for p in prior_front)
                result["status"] = "paired, within prior frontier" if already_covered else "paired, extended prior frontier"
            else:
                result["status"] = "remeasured in another campaign round"
                result["viability_change_pp"] = row[VIABILITY] - target[VIABILITY]
                result["force_change_N_per_needle"] = row[FORCE] - target[FORCE]
        resolutions[kind] = result
    return resolutions


def register_questions(ledger, agenda, front):
    for kind, target in agenda["requests"].items():
        key = (kind, target["formulation_id"], target["batch_id"])
        ledger.setdefault(key, {"kind": kind, "target": target, "opened_after": agenda["evidence_through"],
                                "prior_front": [{endpoint: p[endpoint] for endpoint in (VIABILITY, FORCE)} for p in front],
                                "status": "open", "answered_in": None})


def update_questions(ledger, observed):
    on_slate = set(observed.formulation_id)
    for entry in ledger.values():
        if entry["answered_in"] or entry["target"]["formulation_id"] not in on_slate:
            continue
        kind = entry["kind"]
        request = {"requests": {kind: {"target": entry["target"]}}}
        answer = resolve_requests(request, observed, entry["prior_front"])[kind]
        entry["status"] = answer["status"]
        if answer["status"] not in {"formation missing", "paired outcome missing"}:
            entry["answered_in"] = observed.batch_id.iloc[0]


def forecast_scores(frame, endpoint, forecast, sd):
    pairs = frame.loc[~frame.control].dropna(subset=[endpoint, forecast])
    error = pairs[forecast] - pairs[endpoint]
    rho = None
    if len(pairs) >= 2 and min(pairs[forecast].nunique(), pairs[endpoint].nunique()) > 1:
        rho = float(pairs[forecast].rank().corr(pairs[endpoint].rank()))
    bands = pairs.loc[pairs[sd].gt(0)]
    width = 2 * 1.96 * bands[sd]
    outside = ((bands[forecast] - bands[endpoint]).abs() - width / 2).clip(lower=0)
    return {"n": len(pairs), "rho": rho, "mae": float(error.abs().mean()) if len(error) else None,
            "interval_n": len(bands), "interval_hits": int(outside.eq(0).sum()),
            "interval_width": float(width.mean()) if len(bands) else None,
            "interval_score": float((width + 40 * outside).mean()) if len(bands) else None}


def score_shortlist(plan, observed):
    eligible = observed.set_index("formulation_id").loc[plan["rankable_ids"]]
    result = {"slots": len(plan["shortlist"]), "rankable": len(eligible), "observed": int(eligible[VIABILITY].count()),
              "selected_mean": None, "random_mean": None, "recovered": None}
    if len(eligible) >= 3 and eligible[VIABILITY].notna().all():
        selected = eligible.loc[plan["shortlist"], VIABILITY]
        actual = eligible.sort_values([VIABILITY, "selection_rank"], ascending=[False, True]).head(3)
        result.update(selected_mean=float(selected.mean()), random_mean=float(eligible[VIABILITY].mean()),
                      recovered=len(set(selected.index) & set(actual.index)))
    return result


def replay(archive=ARCHIVE, first=13, through=None, external=None, agent_runner=None, agent_round=None, review_record=None):
    if first < 1 or (through is not None and through < first - 1):
        raise ValueError("The cutoff must include the starting round or its immediate predecessor")
    if external and (external[0] < first or (through is not None and external[0] > through)):
        raise ValueError("The supplied receipt is outside the requested evidence window")
    if review_record is not None:
        if agent_runner or agent_round is not None:
            raise ValueError("A recorded review supplies its own agent round and answers")
        agent_round = int(review_record["round"].removeprefix("ROUND_"))
    if agent_round is not None and ((not agent_runner and review_record is None) or agent_round <= first or (through is not None and agent_round > through + 1)):
        raise ValueError("Choose an agent round with earlier replay evidence and supply an agent runner")
    batches, observed_frames, front, ledger = [], [], [], {}
    agenda = {"evidence_through": None, "focus": "establish the measured two-endpoint trade-off",
              "requests": {}, "reserve_confirmation": False}
    number, pending, used_external, reviewed_round = first, None, False, None
    while True:
        directory = Path(archive) / f"ROUND_{number:03}"
        if not directory.is_dir():
            if not batches or (through is not None and number <= through):
                raise ValueError(f"Missing proposal for {directory.name}")
            break
        proposal, digest = frozen_slate(directory)
        if (agent_runner or review_record is not None) and batches and (agent_round is None or number == agent_round):
            # The caller's local model only receives evidence available BEFORE this round's results.
            if __package__:
                from .deliberation import deliberate, packet_digest, recorded_review
            else:
                from deliberation import deliberate, packet_digest, recorded_review
            evidence = {k: dict(v) for k, v in agenda["requests"].items()}
            for target in evidence.values():
                prior = history.loc[history.formulation_id.eq(target["formulation_id"])]
                target["prior_measurements"] = records(prior[["batch_id", VIABILITY, FORCE, GATE, "preparation_id_count", "cell_batch_id_count"]])
            previous = batches[-1]
            evidence["last_round"] = {key: previous[key] for key in ("intact", "gate_measured", "primary_intact", "primary_gate_measured", "viability", "force", "hypervolume")}
            reference = observed_frames[-1].loc[observed_frames[-1].control, ["batch_id", VIABILITY, FORCE, GATE]]
            evidence["last_round"]["reference_measurements"] = records(reference)
            decisions = ["keep_default"] + (["release_confirmation_slot"] if agenda["reserve_confirmation"] else [])
            packet = {"goal": "Maximize MSC post-thaw viability and comparable same-batch terminal force among intact patches.",
                      "evidence_through": agenda["evidence_through"], "round": directory.name,
                      "eligible_decisions": decisions, "reference_ids": list(evidence), "evidence": evidence,
                      "default_decision": "keep_default", "policy": POLICY, "default_focus": agenda["focus"],
                      "screen_budget": len(proposal), "mechanical_budget": int(proposal.primary.sum()),
                      "endpoint_units": {VIABILITY: "%", FORCE: "N per loaded needle"},
                      "application_minima": None}
            review = recorded_review(packet, review_record) if review_record is not None else deliberate(packet, agent_runner)
            reviewed_round = number
            review["packet_sha256"] = packet_digest(packet)
            if review["decision"] == "release_confirmation_slot":
                agenda = dict(agenda, reserve_confirmation=False, focus="retain the frontier repeat as an open question and return its slot to the optimizer")
        else:
            review = None
        plan = freeze_plan(proposal, digest, agenda)
        if review is not None:
            plan["agent_review"] = review
        if through is not None and number > through:
            pending = plan
            break
        receipt = directory / "completed/ingestion_commit.json"
        if external and number == external[0]:
            packet = Path(external[1])
            if not packet.is_file():
                raise ValueError(f"Supplied receipt does not exist: {packet}")
            if receipt.is_file() and packet.read_bytes() != receipt.read_bytes():
                raise ValueError(f"Supplied receipt conflicts with the archived {directory.name} receipt")
            receipt, used_external = packet, True
        if not receipt.is_file():
            if through is not None and number <= through:
                raise ValueError(f"{directory.name} has no validated result receipt")
            pending = plan
            break
        observed, receipt_hash = open_results(receipt, directory, proposal, digest)
        answers = resolve_requests(plan, observed, front)
        update_questions(ledger, observed)
        observed_frames.append(observed)
        history = pd.concat(observed_frames, ignore_index=True)
        frontier, pair_count, area = pareto_evidence(history)
        front = records(frontier[POINT])
        agenda = next_agenda(observed, history, frontier)
        register_questions(ledger, agenda, front)
        primary = observed.loc[observed.primary, GATE]
        batches.append({"plan": plan, "receipt_sha256": receipt_hash, "resolutions": answers,
                        "allocation_evidence": allocation_evidence(plan, observed),
                        "shortlist_score": score_shortlist(plan, observed),
                        "viability": forecast_scores(observed, VIABILITY, "viability_forecast", "viability_std"),
                        "force": forecast_scores(observed, FORCE, "force_forecast", "critical_axial_load_std"),
                        "intact": int(observed[GATE].eq(1).sum()), "gate_measured": int(observed[GATE].count()),
                        "primary_intact": int(primary.eq(1).sum()), "primary_gate_measured": int(primary.count()),
                        "cumulative_pairs": pair_count, "frontier_size": len(front), "hypervolume": area,
                        "next_agenda": agenda})
        number += 1
    if external and not used_external:
        raise ValueError("The supplied receipt could not be reached in chronological order")
    if agent_round is not None and reviewed_round != agent_round:
        raise ValueError("The selected agent round has no available frozen proposal")
    totals, paired_points = {}, []
    if observed_frames:
        totals = {"viability": forecast_scores(history, VIABILITY, "viability_forecast", "viability_std"),
                  "force": forecast_scores(history, FORCE, "force_forecast", "critical_axial_load_std")}
        measured = history.loc[~history.control & history[GATE].eq(1)].dropna(subset=[VIABILITY, FORCE])
        paired_points = records(measured[POINT])
    return {"policy": POLICY, "first_round": first, "rounds": batches, "totals": totals, "paired_points": paired_points,
            "frontier": front, "question_ledger": list(ledger.values()), "next_agenda": agenda, "pending_plan": pending}


if __name__ == "__main__":
    if __package__:
        from .render import brief, plot
    else:
        from render import brief, plot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE, help="Round directories from a v2-compatible project")
    parser.add_argument("--first", type=int, default=13, help="First full-mechanics round in the replay")
    parser.add_argument("--through", type=int, help="Last result to reveal; display the following frozen plan if present")
    parser.add_argument("--receipt", nargs=2, metavar=("ROUND_NUMBER", "PATH"), help="Score one supplied validated receipt")
    parser.add_argument("--agent-runner", help="Opt into a scientist and critic via 'codex' or a trusted local JSON executable")
    parser.add_argument("--agent-round", type=int, help="Limit model calls to this planning round and use the default policy elsewhere")
    parser.add_argument("--recorded-review", type=Path, help="Replay a recorded scientist and critic exchange on its matching packet")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable decisions and source hashes")
    parser.add_argument("--figure", type=Path, help="Write the historical comparison figure to this path")
    args = parser.parse_args()
    try:
        external = (int(args.receipt[0]), args.receipt[1]) if args.receipt else None
        record = json.loads(args.recorded_review.read_text()) if args.recorded_review else None
        result = replay(args.archive, args.first, args.through, external, args.agent_runner, args.agent_round, record)
        if args.figure:
            plot(result, args.figure)
        print(json.dumps(result, indent=2, allow_nan=False) if args.json else brief(result))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Round review stopped: {error}\n")
