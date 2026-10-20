#!/usr/bin/env python3
"""Explain v2 endpoint models without changing observations, proposals or selection.

Default: the model cutoff for the next round after the latest completed round.
--all-rounds: every completed round at its preceding cutoff, plus the current model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from helper.explainability import ExplanationSettings, explain_round, load_saved_reports, write_history, write_presence_history
from helper.registry import load_registry
from helper.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--round", dest="round_id", help="Explain models before this round, e.g. ROUND_009.")
    selection.add_argument("--all-rounds", action="store_true", help="Backfill completed rounds and explain the current model.")
    selection.add_argument("--history-only", action="store_true", help="Redraw signed history from saved SHAP summaries without refitting models.")
    selection.add_argument("--presence-history-only", action="store_true", help="Compare signed SHAP only where each ingredient is present, with counts; no refitting.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Default: results/multi_objective_v2/reports/explainability")
    parser.add_argument("--max-explain", type=int, default=48)
    parser.add_argument("--max-background", type=int, default=16)
    parser.add_argument("--permutations", type=int, default=16)
    parser.add_argument("--top-features", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir or root / "results/multi_objective_v2/reports/explainability"
    settings = ExplanationSettings(args.max_explain, args.max_background, args.permutations, args.top_features, args.seed)
    registry = load_registry(root / "config_v2/ingredients.yaml")
    if args.presence_history_only:
        write_presence_history(load_saved_reports(output, include_shap_values=True), output, registry)
        print(f"Wrote presence-conditioned SHAP histories from saved rows in {output.resolve()}", flush=True)
        return
    if args.history_only:
        write_history(load_saved_reports(output), output, registry, settings)
        print(f"Redrew signed history from saved SHAP summaries in {output.resolve()}", flush=True)
        return
    forms = pd.read_csv(root / "data/processed_v2/formulations.csv")
    observations = pd.read_csv(root / "data/processed_v2/observations.csv")
    completed = sorted(int(path.parents[1].name.split("_")[-1]) for path in
                       (root / "results/multi_objective_v2/rounds").glob("ROUND_*/completed/completed.csv"))
    current = max(completed, default=0) + 1
    if args.round_id:
        try:
            target = int(args.round_id.removeprefix("ROUND_"))
        except ValueError:
            parser.error("--round must be an integer or ROUND_NNN")
        if target not in completed and target != current:
            parser.error("--round must name a completed round or the current next round")
        targets = [target]
    else:
        targets = completed + [current] if args.all_rounds else [current]
    reports = []
    for target in targets:
        destination = output / ("current" if target == current else f"ROUND_{target:03d}")
        print(f"Explaining model before Round {target} (data through Round {target - 1})...", flush=True)
        report = explain_round(root, forms, observations, registry, target, destination, settings)
        reports.append(report)
        print(f"  Wrote {destination}", flush=True)
    if args.all_rounds:
        write_history(reports, output, registry, settings)
        write_presence_history(reports, output, registry)
    print(f"Reports written to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
