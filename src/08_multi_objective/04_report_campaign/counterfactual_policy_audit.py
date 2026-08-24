#!/usr/bin/env python3
"""Write a read-only Round 3-6 audit of the forward-only Round-6 policies."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.candidates import unavailable_features_from_config
from helper.cold_start import (
    annotate_cold_start_candidates,
    build_cold_start_context,
    resolve_cold_start_policy,
)
from helper.config import load_availability_config, load_optimization_config
from helper.endpoints import aggregate_intact_patch_replicates
from helper.intact_policy import (
    annotate_intact_combination_evidence,
    build_intact_evidence,
    resolve_intact_combination_policy,
)
from helper.paths import (
    AVAILABILITY_CONFIG,
    FORMULATIONS_PATH,
    OBSERVATIONS_PATH,
    RESULTS_V2_DIR,
)
from helper.registry import load_registry, presence_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulations", default=str(FORMULATIONS_PATH))
    parser.add_argument("--observations", default=str(OBSERVATIONS_PATH))
    parser.add_argument("--availability-config", default=str(AVAILABILITY_CONFIG))
    parser.add_argument("--results-root", default=str(RESULTS_V2_DIR))
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_V2_DIR / "reports" / "round6_policy_audit"),
    )
    return parser.parse_args()


def _observed_round_results(
    observations: pd.DataFrame,
    batch_id: str,
) -> pd.DataFrame:
    batch = observations.loc[
        observations["batch_id"].fillna("").astype(str).eq(batch_id)
        & observations["source_type"].fillna("").astype(str).eq("wetlab_feedback")
    ].copy()
    if batch.empty:
        return pd.DataFrame(
            columns=[
                "formulation_id",
                "observed_viability_percent",
                "observed_intact_patch_pass",
            ]
        )
    viability = (
        batch.loc[batch["endpoint"].astype(str).eq("viability_percent")]
        .groupby("formulation_id", as_index=False)
        .agg(observed_viability_percent=("value", "mean"))
    )
    intact = (
        batch.loc[
            batch["endpoint"].astype(str).eq("intact_patch_formation_pass")
        ]
        .groupby("formulation_id", as_index=False)
        .agg(observed_intact_patch_pass=("value", aggregate_intact_patch_replicates))
    )
    return viability.merge(intact, on="formulation_id", how="outer")


def _ingredient_present(frame: pd.DataFrame, feature_name: str) -> pd.Series:
    return (
        pd.to_numeric(frame.get(feature_name, 0.0), errors="coerce")
        .fillna(0.0)
        .abs()
        >= presence_threshold(feature_name)
    )


def main() -> None:
    args = parse_args()
    registry = load_registry()
    config = load_optimization_config()
    formulations = pd.read_csv(args.formulations)
    observations = pd.read_csv(args.observations)
    unavailable = unavailable_features_from_config(
        load_availability_config(args.availability_config), registry
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_frames: list[pd.DataFrame] = []
    round_summaries: list[dict[str, object]] = []
    results_root = Path(args.results_root)
    for round_number in range(3, 7):
        batch_id = f"ROUND_{round_number:03d}"
        proposal_path = results_root / "rounds" / batch_id / "proposal" / "proposal.csv"
        if not proposal_path.exists():
            round_summaries.append(
                {"batch_id": batch_id, "status": "proposal_missing"}
            )
            continue
        proposal = pd.read_csv(proposal_path)
        intact_policy = replace(
            resolve_intact_combination_policy(config, round_number), active=True
        )
        intact_evidence = build_intact_evidence(
            formulations,
            observations,
            registry,
            intact_policy,
            round_number,
        )
        audited = annotate_intact_combination_evidence(
            proposal,
            intact_evidence,
            registry,
            intact_policy,
        )
        cold_policy = replace(
            resolve_cold_start_policy(config, round_number), active=True
        )
        cold_context = build_cold_start_context(
            formulations,
            observations,
            registry,
            cold_policy,
            round_number,
            unavailable_feature_names=unavailable,
        )
        audited = annotate_cold_start_candidates(audited, registry, cold_context)
        observed = _observed_round_results(observations, batch_id)
        audited = audited.merge(observed, on="formulation_id", how="left")
        audited.insert(0, "audit_batch_id", batch_id)
        audited["audit_policy_applied_counterfactually"] = round_number < 6
        audited["ordinary_row_for_cold_cap"] = ~audited.get(
            "candidate_origin", pd.Series("", index=audited.index)
        ).astype(str).isin(cold_policy.exempt_candidate_origins)

        ordinary_cold_counts: dict[str, int] = {}
        for feature_name in cold_context.cold_ingredients:
            ordinary_cold_counts[feature_name] = int(
                (
                    audited["ordinary_row_for_cold_cap"]
                    & _ingredient_present(audited, feature_name)
                ).sum()
            )
        round_summaries.append(
            {
                "batch_id": batch_id,
                "status": "audited",
                "cold_ingredients": list(cold_context.cold_ingredients),
                "prior_distinct_formulation_counts": {
                    feature_name: cold_context.evidence_counts[feature_name]
                    for feature_name in cold_context.cold_ingredients
                },
                "ordinary_cold_counts": ordinary_cold_counts,
                "cold_cap_violations": {
                    feature_name: count
                    for feature_name, count in ordinary_cold_counts.items()
                    if count > cold_policy.max_ordinary_rows_per_ingredient
                },
                "rows_with_empirical_intact_deduction": int(
                    (audited["intact_combination_screening_penalty"] > 0).sum()
                ),
                "maximum_empirical_intact_deduction": float(
                    audited["intact_combination_screening_penalty"].max()
                ),
            }
        )
        detail_frames.append(audited)

    detail = pd.concat(detail_frames, ignore_index=True, sort=False)
    detail.to_csv(output_dir / "round3_to_round6_policy_audit.csv", index=False)
    (output_dir / "round3_to_round6_policy_audit.json").write_text(
        json.dumps(
            {
                "audit_version": "round6_policy_counterfactual_v1",
                "historical_artifacts_modified": False,
                "replacement_outcomes_are_unknown": True,
                "rounds": round_summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Round 3-6 Forward-Policy Audit",
        "",
        "This is a counterfactual audit only. Rounds 1-5 and their frozen proposals were not changed.",
        "The audit identifies positions the new rules would have constrained; it does not claim that unknown replacements would have performed better.",
        "",
        "| Round | Cold ingredients under reconstructed prior evidence | Ordinary cold-cap violations | Rows with intact deduction | Largest deduction |",
        "|---|---|---|---:|---:|",
    ]
    for summary in round_summaries:
        if summary.get("status") != "audited":
            lines.append(f"| {summary['batch_id']} | proposal missing | — | — | — |")
            continue
        cold = ", ".join(summary["cold_ingredients"]) or "none"
        violations = ", ".join(
            f"{feature}={count}"
            for feature, count in summary["cold_cap_violations"].items()
        ) or "none"
        lines.append(
            f"| {summary['batch_id']} | {cold} | {violations} | "
            f"{summary['rows_with_empirical_intact_deduction']} | "
            f"{summary['maximum_empirical_intact_deduction']:.4f} |"
        )

    for batch_id, feature_name, display_name in (
        ("ROUND_004", "raffinose_M", "raffinose"),
        ("ROUND_005", "acetamide_M", "acetamide"),
    ):
        rows = detail.loc[detail["audit_batch_id"].eq(batch_id)].copy()
        if rows.empty or "observed_viability_percent" not in rows.columns:
            continue
        present = _ingredient_present(rows, feature_name)
        with_mean = pd.to_numeric(
            rows.loc[present, "observed_viability_percent"], errors="coerce"
        ).mean()
        without_mean = pd.to_numeric(
            rows.loc[~present, "observed_viability_percent"], errors="coerce"
        ).mean()
        lines.extend(
            [
                "",
                f"- {batch_id}: {int(present.sum())} {display_name}-containing rows averaged {with_mean:.2f}% observed viability; the other {int((~present).sum())} averaged {without_mean:.2f}%. The independent cap would retain at most two ordinary {display_name} rows and free three positions.",
            ]
        )
    round3_ordinary = detail.loc[
        detail["audit_batch_id"].eq("ROUND_003")
        & ~detail.get("candidate_origin", pd.Series("", index=detail.index))
        .astype(str)
        .isin(("retest", "rescue_dilution"))
        & detail["intact_combination_screening_penalty"].gt(0)
    ].sort_values("selection_rank")
    if not round3_ordinary.empty:
        row = round3_ordinary.iloc[0]
        lines.extend(
            [
                "",
                "- ROUND_003 intact counterfactual: ordinary candidate "
                f"`{row['candidate_id']}` would have received a "
                f"{float(row['intact_combination_screening_penalty']):.3f} "
                "screening deduction and subsequently failed the intact gate.",
            ]
        )
    round5_mixed = detail.loc[
        detail["audit_batch_id"].eq("ROUND_005")
        & detail["empirical_combination_weighted_passes"].gt(0)
        & detail["empirical_combination_weighted_failures"].gt(0)
        & detail["intact_combination_screening_penalty"].eq(0)
        & detail["observed_intact_patch_pass"].eq(1)
    ].sort_values("selection_rank")
    if not round5_mixed.empty:
        row = round5_mixed.iloc[0]
        lines.extend(
            [
                "",
                "- ROUND_005 intact counterfactual: candidate "
                f"`{row['candidate_id']}` had closer/stronger weighted pass "
                "than failure evidence, would have received no deduction, and "
                "subsequently passed intact.",
            ]
        )
    lines.extend(
        [
            "",
            "The detailed CSV contains every proposal row, reconstructed empirical intact evidence, cold-start status, and observed outcome where available.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Round 3-6 policy audit: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
