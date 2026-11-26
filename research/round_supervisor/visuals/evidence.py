"""Build the scientific figure inputs from the archived, as-of-R19 CryoMN replay."""

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import supervisor as sup
from helper.config import load_optimization_config
from helper.similarity import SimilarityIndex, resolve_similarity_policy


ROOT = sup.ROOT
TABLES = ROOT / "results/multi_objective_v2/reports/prospective/tables"
HERE = Path(__file__).resolve().parent
ROUNDS = list(range(13, 20))
V, F, G = sup.VIABILITY, sup.FORCE, sup.GATE


def historical_predictions(evaluated, observations):
    history = observations.loc[observations.endpoint.eq(V) & observations.batch_id.str.fullmatch(r"ROUND_\d+", na=False)
                               & ~observations.formulation_id.str.startswith("reference_") & observations.value.notna()]
    history = history.groupby(["batch_id", "formulation_id"], as_index=False).value.mean()
    history["round_number"] = history.batch_id.str[6:].astype(int)
    registry = sup.REGISTRY
    config = load_optimization_config()
    similarity = SimilarityIndex(registry, resolve_similarity_policy(config, max(ROUNDS)))
    forms = pd.read_csv(ROOT / "data/processed_v2/formulations.csv").set_index("formulation_id", verify_integrity=True)
    ids = sorted(set(evaluated.formulation_id) | set(history.formulation_id))
    coords = pd.DataFrame([similarity.prepare(forms.loc[k])[1] for k in ids], index=ids)
    result = evaluated.copy()
    for n, current in result.groupby("round_number"):
        before = history.loc[history.round_number.lt(n)].sort_values("round_number")
        by_recipe = before.groupby("formulation_id").value.mean()
        latest = before.groupby("formulation_id").value.last()
        coord = coords.loc[by_recipe.index].to_numpy()
        result.loc[current.index, "earlier_mean"] = before.value.mean()
        for ix, fid in current.formulation_id.items():
            if fid in latest.index:
                predicted = latest.at[fid]
            else:
                distances = np.linalg.norm(coord - coords.loc[fid].to_numpy(), axis=1)
                near = np.argsort(distances, kind="stable")[:config["retest"]["nearest_neighbor_count"]]
                predicted = by_recipe.iloc[near].mean()
            result.loc[ix, "similar_recipes"] = predicted
    return result


def load():
    review = json.loads((HERE.parent / "round20_agent_review.json").read_text())
    replay = sup.replay(through=19, review_record=review)
    runs = {int(r["plan"]["batch_id"][-3:]): r for r in replay["rounds"]}
    archive = pd.read_csv(TABLES / "prospective_evaluation_table.csv")
    all_cases = archive.loc[archive.round_number.isin(ROUNDS) & archive.active_phase.eq("mechanics_enabled")].copy()
    if all_cases.duplicated(["round_number", "formulation_id", "endpoint"]).any():
        raise ValueError("Multiple archive rows for a formulation, round and endpoint")
    formal = all_cases.loc[all_cases.formal_metric_eligible]
    raw = pd.read_csv(ROOT / "data/processed_v2/observations.csv", low_memory=False)
    v = historical_predictions(formal[formal.endpoint.eq(V)], raw)
    f = formal[formal.endpoint.eq(F) & formal.mechanical_definition_id.eq(sup.DEFINITION)].copy()
    slates = {n: sup.frozen_slate(sup.ARCHIVE / f"ROUND_{n:03}")[0] for n in range(13, 21)}
    intervals = pd.read_csv(TABLES / "interval_scores.csv")
    intervals = intervals[intervals.scope.eq("round")].copy()
    intervals["n_round"] = intervals.round_id.str[6:].astype(int)
    intervals = intervals[intervals.n_round.isin(ROUNDS)]
    reference = pd.read_csv(TABLES / "reference_monitoring.csv")
    reference["n_round"] = reference.batch_id.str[6:].astype(int)
    reference = reference[reference.n_round.isin(ROUNDS) & reference.endpoint.eq(V)].sort_values("n_round")
    outcomes = all_cases.pivot(index=["round_number", "formulation_id"], columns="endpoint", values="observed_mean")
    sample_rows, matched, lottery, ranks = [], [], {}, []
    for n, r in runs.items():
        hist = slates[n]
        measured = outcomes.loc[n]
        for row in hist.itertuples():
            o = measured.loc[row.formulation_id]
            if o[G] not in (0, 1) or pd.isna(o[V]):
                raise ValueError(f"Assay-coverage display requires observed formation and viability for {n}, {row.formulation_id}")
            status = "control" if row.control else ("gate failed" if o[G] == 0 else "paired" if pd.notna(o[F]) else "screen only")
            sample_rows.append((n, row.selection_rank, status, row.primary))
        out_counts = {"paired": 0, "failed_gate": 0, "unresolved": 0}
        for slot, choice in enumerate(r["plan"]["new_project_draft"]["compression_priorities"], 1):
            key = choice["formulation_id"]
            status = "unresolved"
            if key in measured.index:
                o = measured.loc[key]
                if o[G] == 0:
                    status = "failed_gate"
                elif o[G] == 1 and pd.notna(o[V]) and pd.notna(o[F]):
                    status = "paired"
            kept = key in r["plan"]["primary_ids"]
            matched.append((n, slot, choice["purpose"], status, kept))
            out_counts[status] += 1
        if any(r["allocation_evidence"][key] != value for key, value in out_counts.items()):
            raise ValueError(f"Nomination counts differ in R{n}")
        vv = v[v.round_number.eq(n)].sort_values(["observed_mean", "selection_rank"], ascending=[False, True])
        vv = vv.assign(outcome_rank=range(1, len(vv) + 1)).set_index("formulation_id")
        top = vv.loc[r["plan"]["shortlist"]]
        ranks.append(top.outcome_rank.to_list())
        lottery[n] = np.asarray([np.mean(trio) for trio in combinations(vv.observed_mean, 3)])
        s = r["shortlist_score"]
        if not (np.isclose(lottery[n].mean(), s["random_mean"]) and np.isclose(top.observed_mean.mean(), s["selected_mean"])):
            raise ValueError(f"Random-triplet or shortlist reference differs in R{n}")
    if len(v) != replay["totals"]["viability"]["n"] or len(f) != replay["totals"]["force"]["n"]:
        raise ValueError("Forecast cohort does not reconcile with replay")
    pts = pd.DataFrame(replay["paired_points"])
    pts["n_round"] = pts.batch_id.str[6:].astype(int)
    leader = replay["pending_plan"]["requests"]["repeat_frontier"]["target"]["formulation_id"]
    return dict(replay=replay, runs=runs, v=v, f=f, intervals=intervals, reference=reference, slates=slates,
                outcomes=outcomes, raw=raw, points=pts, frontier=pd.DataFrame(replay["frontier"]), leader=leader,
                samples=pd.DataFrame(sample_rows, columns=["round", "rank", "status", "primary"]),
                nominations=pd.DataFrame(matched, columns=["round", "slot", "purpose", "status", "kept"]),
                lottery=lottery, ranks=np.array(ranks))
