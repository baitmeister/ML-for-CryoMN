"""Prepare plotting evidence without fitting models or changing campaign state."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from .registry import load_registry
from .endpoints import parse_bool

def _round_number(value: object) -> int | None:
    text = str(value).strip()
    if text.startswith("ROUND_") and text.removeprefix("ROUND_").isdigit():
        return int(text.removeprefix("ROUND_"))
    return None

def _wilson_interval(passes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    proportion = passes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)

def _short_formulation(row: pd.Series) -> str:
    registry = load_registry()
    parts: list[str] = []
    for feature in registry.feature_names:
        value = pd.to_numeric(row.get(feature), errors="coerce")
        if pd.isna(value) or float(value) <= 0:
            continue
        name = registry.get_by_feature(feature).display_name
        parts.append(name)
    if not parts:
        return "no active ingredients"
    if len(parts) <= 3:
        return " + ".join(parts)
    return " + ".join(parts[:3]) + f" +{len(parts) - 3}"

def _formal_viability(prospective: pd.DataFrame) -> pd.DataFrame:
    mask = (
        prospective["endpoint"].astype(str).eq("viability_percent")
        & prospective["formal_metric_eligible"].map(lambda value: parse_bool(value) is True)
    )
    result = prospective.loc[mask].copy()
    for column in (
        "prediction_mean",
        "prediction_std",
        "observed_mean",
        "absolute_error",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["prediction_mean", "observed_mean"])

def _timeline_table(
    observations: pd.DataFrame,
    prospective: pd.DataFrame,
    metrics: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    observations = observations.reindex(columns=list(dict.fromkeys([*observations.columns,'batch_id','endpoint','value','formulation_id'])))
    prospective = prospective.reindex(columns=list(dict.fromkeys([*prospective.columns,'round_id','active_phase','provenance_class'])))
    metrics = metrics.reindex(columns=list(dict.fromkeys([*metrics.columns,'scope','round_id','endpoint'])))
    completed_rounds = sorted(
        {
            str(value)
            for value in observations.get("batch_id", pd.Series(dtype=str))
            if _round_number(value) is not None
        },
        key=lambda value: int(value.removeprefix("ROUND_")),
    )
    proposed_rounds = sorted(
        {
            str(value)
            for value in candidates.get("batch_id", pd.Series(dtype=str))
            if _round_number(value) is not None
        },
        key=lambda value: int(value.removeprefix("ROUND_")),
    )
    rounds = sorted(
        set(completed_rounds) | set(proposed_rounds),
        key=lambda value: int(value.removeprefix("ROUND_")),
    )
    rows: list[dict[str, object]] = []
    round_metrics = metrics.loc[metrics["scope"].astype(str).eq("round")]
    for round_id in rounds:
        number = int(round_id.removeprefix("ROUND_"))
        round_obs = observations.loc[observations["batch_id"].astype(str).eq(round_id)].copy()
        phase_values = prospective.loc[
            prospective["round_id"].astype(str).eq(round_id), "active_phase"
        ].dropna()
        phase = str(phase_values.iloc[0]) if not phase_values.empty else "unrecorded"
        provenance_values = prospective.loc[
            prospective["round_id"].astype(str).eq(round_id), "provenance_class"
        ].dropna()
        provenance = str(provenance_values.iloc[0]) if not provenance_values.empty else ("current_proposal" if round_id in proposed_rounds else "unrecorded")
        viability = round_obs.loc[
            round_obs["endpoint"].astype(str).eq("viability_percent")
        ].copy()
        if not viability.empty:
            viability["value"] = pd.to_numeric(viability["value"], errors="coerce")
            grouped = viability.groupby("formulation_id")["value"].mean().dropna()
            for formulation_id, value in grouped.items():
                rows.append(
                    {
                        "round_id": round_id,
                        "round_number": number,
                        "phase": phase,
                        "provenance": provenance,
                        "panel": "observed_viability",
                        "series": "individual_formulation",
                        "formulation_id": formulation_id,
                        "value": float(value),
                        "lower": np.nan,
                        "upper": np.nan,
                        "count_pass": np.nan,
                        "count_fail": np.nan,
                    }
                )
            rows.append(
                {
                    "round_id": round_id,
                    "round_number": number,
                    "phase": phase,
                    "provenance": provenance,
                    "panel": "observed_viability",
                    "series": "median_iqr",
                    "formulation_id": "",
                    "value": float(grouped.median()),
                    "lower": float(grouped.quantile(0.25)),
                    "upper": float(grouped.quantile(0.75)),
                    "count_pass": np.nan,
                    "count_fail": np.nan,
                }
            )
        for metric_name in ("mae", "bias", "interval_95_coverage"):
            match = round_metrics.loc[
                round_metrics["round_id"].astype(str).eq(round_id)
                & round_metrics["endpoint"].astype(str).eq("viability_percent")
            ]
            value = pd.to_numeric(match.iloc[0].get(metric_name), errors="coerce") if not match.empty else np.nan
            rows.append(
                {
                    "round_id": round_id,
                    "round_number": number,
                    "phase": phase,
                    "provenance": provenance,
                    "panel": "prospective_performance",
                    "series": metric_name,
                    "formulation_id": "",
                    "value": value,
                    "lower": np.nan,
                    "upper": np.nan,
                    "count_pass": np.nan,
                    "count_fail": np.nan,
                }
            )
        intact = round_obs.loc[
            round_obs["endpoint"].astype(str).eq("intact_patch_formation_pass")
        ].copy()
        if intact.empty:
            passes = failures = total = 0
            proportion = low = high = np.nan
        else:
            intact["value"] = pd.to_numeric(intact["value"], errors="coerce")
            aggregated = intact.groupby("formulation_id")["value"].min().dropna()
            passes = int(aggregated.ge(0.5).sum())
            failures = int(aggregated.lt(0.5).sum())
            total = passes + failures
            proportion = passes / total if total else np.nan
            low, high = _wilson_interval(passes, total)
        rows.append(
            {
                "round_id": round_id,
                "round_number": number,
                "phase": phase,
                "provenance": provenance,
                "panel": "feasibility_mechanics",
                "series": "intact_pass_proportion",
                "formulation_id": "",
                "value": proportion,
                "lower": low,
                "upper": high,
                "count_pass": passes,
                "count_fail": failures,
            }
        )
        mechanical_count = int(
            round_obs["endpoint"].astype(str).eq(
                "critical_axial_load_N_per_needle"
            ).sum()
        )
        rows.append(
            {
                "round_id": round_id,
                "round_number": number,
                "phase": phase,
                "provenance": provenance,
                "panel": "feasibility_mechanics",
                "series": "mechanical_measurement_count",
                "formulation_id": "",
                "value": mechanical_count,
                "lower": np.nan,
                "upper": np.nan,
                "count_pass": np.nan,
                "count_fail": np.nan,
            }
        )
    return pd.DataFrame(rows, columns=["round_id", "round_number", "phase", "provenance", "panel", "series", "formulation_id", "value", "lower", "upper", "count_pass", "count_fail"])

def _trust_table(prospective: pd.DataFrame) -> pd.DataFrame:
    viability = _formal_viability(prospective)
    rows: list[dict[str, object]] = []
    for _, row in viability.iterrows():
        rows.append(
            {
                "panel": "viability_prediction",
                "candidate_id": row["candidate_id"],
                "round_id": row["round_id"],
                "prediction": row["prediction_mean"],
                "observed": row["observed_mean"],
                "uncertainty_std": row["prediction_std"],
                "absolute_error": row["absolute_error"],
                "nominal_coverage": np.nan,
                "empirical_coverage": np.nan,
                "interval_width": 3.92 * row["prediction_std"] if pd.notna(row["prediction_std"]) else np.nan,
                "coverage_lower": np.nan,
                "coverage_upper": np.nan,
                "binary_outcome": np.nan,
            }
        )
    valid_std = viability.dropna(subset=["prediction_std"])
    for nominal in (0.50, 0.68, 0.80, 0.90, 0.95):
        z_value = norm.ppf((1.0 + nominal) / 2.0)
        covered = (
            (valid_std["observed_mean"] >= valid_std["prediction_mean"] - z_value * valid_std["prediction_std"])
            & (valid_std["observed_mean"] <= valid_std["prediction_mean"] + z_value * valid_std["prediction_std"])
        )
        covered_count = int(covered.sum())
        coverage_lower, coverage_upper = _wilson_interval(
            covered_count,
            int(len(covered)),
        )
        rows.append(
            {
                "panel": "coverage",
                "candidate_id": "",
                "round_id": "FORMAL_COHORT",
                "prediction": np.nan,
                "observed": np.nan,
                "uncertainty_std": np.nan,
                "absolute_error": np.nan,
                "nominal_coverage": nominal,
                "empirical_coverage": float(covered.mean()) if len(covered) else np.nan,
                "interval_width": float((2 * z_value * valid_std["prediction_std"]).mean()) if len(valid_std) else np.nan,
                "coverage_lower": coverage_lower,
                "coverage_upper": coverage_upper,
                "binary_outcome": np.nan,
            }
        )
    intact = prospective.loc[
        prospective["endpoint"].astype(str).eq("intact_patch_formation_pass")
        & prospective["formal_metric_eligible"].map(lambda value: parse_bool(value) is True)
    ].copy()
    for _, row in intact.iterrows():
        rows.append(
            {
                "panel": "intact_probability",
                "candidate_id": row["candidate_id"],
                "round_id": row["round_id"],
                "prediction": pd.to_numeric(row["prediction_mean"], errors="coerce"),
                "observed": np.nan,
                "uncertainty_std": np.nan,
                "absolute_error": np.nan,
                "nominal_coverage": np.nan,
                "empirical_coverage": np.nan,
                "interval_width": np.nan,
                "coverage_lower": np.nan,
                "coverage_upper": np.nan,
                "binary_outcome": pd.to_numeric(row["observed_mean"], errors="coerce"),
            }
        )
    return pd.DataFrame(rows)

def _decision_table(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    defaults = {"selection_rank": np.nan, "candidate_id": "", "formulation_id": "",
                "viability_prediction_status": "unknown_unrecorded", "predicted_viability_percent": np.nan,
                "viability_std": np.nan, "selection_role": "", "recommendation_type": "",
                "empirical_combination_pass_probability": np.nan, "support_status": "",
                "candidate_origin": "unrecorded", "mechanical_test_recommended": None,
                "mechanical_selection_rank": np.nan, "mechanical_transition_role": ""}
    for column, default in defaults.items():
        if column not in frame: frame[column] = default
    frame = frame.sort_values("selection_rank", kind="stable")
    frame["formulation_description"] = frame.apply(_short_formulation, axis=1)
    status = frame["viability_prediction_status"].fillna("").astype(str)
    reportable = ~status.str.startswith("unknown_")
    frame["public_viability_mean"] = pd.to_numeric(
        frame["predicted_viability_percent"], errors="coerce"
    ).where(reportable)
    frame["public_viability_std"] = pd.to_numeric(
        frame["viability_std"], errors="coerce"
    ).where(reportable)
    columns = [
        "selection_rank",
        "candidate_id",
        "formulation_id",
        "formulation_description",
        "selection_role",
        "recommendation_type",
        "viability_prediction_status",
        "public_viability_mean",
        "public_viability_std",
        "empirical_combination_pass_probability",
        "support_status",
        "candidate_origin",
        "mechanical_test_recommended",
        "mechanical_selection_rank",
        "mechanical_transition_role",
    ]
    return frame[columns].reset_index(drop=True)

def trust_table(prospective):
    prospective=prospective.reindex(columns=list(dict.fromkeys([*prospective.columns,'endpoint','formal_metric_eligible','prediction_mean','prediction_std','observed_mean','absolute_error','candidate_id','round_id'])))
    table=_trust_table(prospective)
    # The legacy constructor returns no columns for a completely empty cohort.
    required=['panel','candidate_id','round_id','prediction','observed','uncertainty_std',
              'absolute_error','nominal_coverage','empirical_coverage','interval_width',
              'coverage_lower','coverage_upper','binary_outcome']
    table=table.reindex(columns=required)
    table['signed_error']=pd.to_numeric(table.prediction,errors='coerce')-pd.to_numeric(table.observed,errors='coerce')
    return table

def decision_table(candidates):
    table=_decision_table(candidates)
    full=(candidates.sort_values('selection_rank', kind='stable') if 'selection_rank' in candidates else candidates).reset_index(drop=True)
    # Include complete input features and IDs in the source table, not just abbreviated labels.
    table=pd.concat([table,full[[col for col in full if col not in table]]],axis=1).copy()
    if 'generation_origin' in table:
        table['selection_origin_group'] = table.candidate_origin
        table['candidate_origin'] = table.generation_origin.fillna(table.candidate_origin)
    table['effective_intact_evidence']=pd.to_numeric(table.get('empirical_combination_weighted_passes', pd.Series(np.nan,index=table.index)),errors='coerce')+pd.to_numeric(table.get('empirical_combination_weighted_failures', pd.Series(np.nan,index=table.index)),errors='coerce')
    table['prior_only']=table.effective_intact_evidence.eq(0)
    ranks=pd.to_numeric(table.mechanical_selection_rank,errors='coerce')
    flags=table.mechanical_test_recommended.map(lambda x:parse_bool(x) is True)
    table['mechanical_assignment']=[f'Primary {int(r)}' if primary and pd.notna(r) else f'Backup {int(r)}' if pd.notna(r) else 'Primary (rank unknown)' if primary else 'Screen only' for r,primary in zip(ranks,flags)]
    reasons={'retest':'Repeat a tested formulation','rescue_dilution':'Dilute a tested formulation',
             'sparse_exploration':'Explore ingredient combinations','local_perturbation':'Explore near an existing formulation',
             'boundary_probe':'Explore higher concentrations',
             'continuous_qlognehvi':'Continuous concentration refinement',
             'finite_pool':'Select from finite pool'}
    unknown_assignment=table.mechanical_test_recommended.map(parse_bool).isna() & ranks.isna()
    table.loc[unknown_assignment,'mechanical_assignment']='Not recorded'
    table['selection_reason']=table.candidate_origin.fillna('unrecorded').map(reasons).fillna(table.candidate_origin.fillna('unrecorded').str.replace('_',' '))
    utility=table.mechanical_transition_role.eq('bootstrap_utility')
    table.loc[utility,'selection_reason']+='; high bootstrap utility'
    return table


timeline_table = _timeline_table
