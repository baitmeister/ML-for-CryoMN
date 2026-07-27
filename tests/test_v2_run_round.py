from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from helper.artifacts import freeze_proposal, round_artifact_paths
from helper.registry import load_registry
from helper.transfer import FORMULATION_BASE_COLUMNS, OBSERVATION_COLUMNS
import run_round as run_round_module
from run_round import (
    NEXT_ROUND_SUMMARY_PATH,
    _resolve_current_summary_path,
    _round_has_new_results,
)


def test_resolve_current_summary_path_prefers_candidate_sibling(tmp_path: Path) -> None:
    candidate_path = tmp_path / "next_round_candidates.csv"
    candidate_path.write_text("formulation_id\n", encoding="utf-8")
    summary_path = tmp_path / "next_round_summary.txt"
    summary_path.write_text("summary\n", encoding="utf-8")

    resolved = _resolve_current_summary_path(candidate_path)

    assert resolved == summary_path


def test_resolve_current_summary_path_falls_back_to_default() -> None:
    resolved = _resolve_current_summary_path("missing_candidates.csv")

    assert resolved == NEXT_ROUND_SUMMARY_PATH


def test_round_has_new_results_false_when_file_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.csv"

    assert _round_has_new_results(missing_path) is False


def test_round_has_new_results_false_when_no_result_columns_filled(tmp_path: Path) -> None:
    candidate_path = tmp_path / "next_round_candidates.csv"
    candidate_path.write_text(
        "formulation_id,candidate_id,viability_percent,intact_patch_formation_pass\n"
        "v2_cand_1,cand_1,,\n"
        "v2_cand_2,cand_2,,\n",
        encoding="utf-8",
    )

    assert _round_has_new_results(candidate_path) is False


def test_round_has_new_results_false_when_result_columns_absent(tmp_path: Path) -> None:
    candidate_path = tmp_path / "next_round_candidates.csv"
    candidate_path.write_text(
        "formulation_id,candidate_id,betaine_M\nv2_cand_1,cand_1,0.2\n",
        encoding="utf-8",
    )

    assert _round_has_new_results(candidate_path) is False


def test_round_has_new_results_true_when_viability_filled(tmp_path: Path) -> None:
    candidate_path = tmp_path / "next_round_candidates.csv"
    candidate_path.write_text(
        "formulation_id,candidate_id,viability_percent,intact_patch_formation_pass\n"
        "v2_cand_1,cand_1,,\n"
        "v2_cand_2,cand_2,72.5,\n",
        encoding="utf-8",
    )

    assert _round_has_new_results(candidate_path) is True


def test_round_has_new_results_true_when_only_instron_file_filled(tmp_path: Path) -> None:
    candidate_path = tmp_path / "next_round_candidates.csv"
    candidate_path.write_text(
        "formulation_id,candidate_id,viability_percent,instron_file\n"
        "v2_cand_1,cand_1,,results/instron_001.csv\n",
        encoding="utf-8",
    )

    assert _round_has_new_results(candidate_path) is True


def test_round_has_new_results_false_when_only_retest_priority_viability_is_prefilled(
    tmp_path: Path,
) -> None:
    """next_round_candidates.csv pre-fills viability_percent on retest_priority
    rows with the formulation's prior observed viability (see helper/retest.py),
    purely as context for the person re-running the test. That carried-over
    value must not be mistaken for a freshly entered wet-lab result, or every
    round containing a retest candidate would be wrongly treated as having
    progressed before any new data was actually entered.
    """
    candidate_path = tmp_path / "next_round_candidates.csv"
    candidate_path.write_text(
        "formulation_id,candidate_id,recommendation_type,viability_percent,intact_patch_formation_pass\n"
        "v2_cand_1,retest_v2_cand_1,retest_priority,26.53,\n"
        "v2_cand_2,rescue_000001,rescue_candidate,,\n"
        "v2_cand_3,cand_3,screening_candidate,,\n",
        encoding="utf-8",
    )

    assert _round_has_new_results(candidate_path) is False


def test_round_has_new_results_true_when_screening_row_viability_is_filled_alongside_retest(
    tmp_path: Path,
) -> None:
    """A real new result on a screening_candidate row must still register as
    progress even when a retest_priority row's carried-over viability is also
    present in the same file.
    """
    candidate_path = tmp_path / "next_round_candidates.csv"
    candidate_path.write_text(
        "formulation_id,candidate_id,recommendation_type,viability_percent\n"
        "v2_cand_1,retest_v2_cand_1,retest_priority,26.53\n"
        "v2_cand_2,cand_2,screening_candidate,71.0\n",
        encoding="utf-8",
    )

    assert _round_has_new_results(candidate_path) is True


def test_run_round_archives_exact_completed_sheet_and_writes_round_reports(
    tmp_path: Path,
) -> None:
    registry = load_registry()
    results_root = tmp_path / "results"
    output_dir = results_root / "next_round"
    output_dir.mkdir(parents=True)
    proposal_path = output_dir / "next_round_candidates.csv"
    proposal_row = {
        "formulation_id": "v2_lifecycle",
        "candidate_id": "cand_lifecycle",
        "selection_rank": 1,
        "recommendation_type": "screening_candidate",
        "mechanical_test_recommended": False,
        "batch_id": "ROUND_002",
        "replicate_id": "",
        "viability_percent": "",
        "intact_patch_formation_pass": "",
        "notes": "",
        "dmso_M": 0.2,
        "active_ingredient_count": 1,
        "predicted_viability_percent": 65.0,
        "viability_std": 10.0,
        "intact_patch_pass_probability": 0.5,
    }
    pd.DataFrame([proposal_row]).to_csv(proposal_path, index=False)
    summary_path = output_dir / "next_round_summary.txt"
    summary_path.write_text("ROUND_002 proposal\n", encoding="utf-8")
    freeze_proposal(
        "ROUND_002",
        proposal_path,
        summary_path=summary_path,
        results_root=results_root,
    )

    filled = pd.DataFrame(
        [
            {
                **proposal_row,
                "replicate_id": "rep_001",
                "viability_percent": 72.5,
                "intact_patch_formation_pass": "yes",
                "notes": "completed lifecycle test",
            }
        ]
    )
    filled.to_csv(proposal_path, index=False)
    filled_bytes = proposal_path.read_bytes()

    formulations_path = tmp_path / "formulations.csv"
    pd.DataFrame(columns=FORMULATION_BASE_COLUMNS + registry.feature_names).to_csv(
        formulations_path,
        index=False,
    )
    observations_path = tmp_path / "observations.csv"
    pd.DataFrame(columns=OBSERVATION_COLUMNS).to_csv(observations_path, index=False)

    subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "src"
                / "08_multi_objective"
                / "03_run_round"
                / "run_round.py"
            ),
            str(proposal_path),
            "--formulations",
            str(formulations_path),
            "--observations",
            str(observations_path),
            "--output-dir",
            str(output_dir),
            "--total-candidate-pool",
            str(results_root / "total_candidate_pool.csv"),
            "--skip-generate",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    artifacts = round_artifact_paths("ROUND_002", results_root)
    assert artifacts.completed_csv.read_bytes() == filled_bytes
    assert (artifacts.reports_dir / "best_performers_summary.txt").exists()
    assert (artifacts.reports_dir / "report_summary.txt").exists()
    assert (artifacts.report_plots_dir / "endpoint_observation_counts.png").exists()
    observations = pd.read_csv(observations_path)
    assert set(observations["endpoint"]) == {
        "viability_percent",
        "intact_patch_formation_pass",
    }


def test_reporting_failure_preserves_ingest_and_blocks_next_slate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = load_registry()
    results_root = tmp_path / "results"
    output_dir = results_root / "next_round"
    output_dir.mkdir(parents=True)
    working_path = output_dir / "next_round_candidates.csv"
    proposal_row = {
        "formulation_id": "v2_report_retry",
        "candidate_id": "cand_report_retry",
        "selection_rank": 1,
        "recommendation_type": "screening_candidate",
        "mechanical_test_recommended": False,
        "batch_id": "ROUND_002",
        "replicate_id": "",
        "viability_percent": "",
        "intact_patch_formation_pass": "",
        "notes": "",
        "dmso_M": 0.2,
        "active_ingredient_count": 1,
        "predicted_viability_percent": 65.0,
        "viability_std": 10.0,
        "intact_patch_pass_probability": 0.5,
    }
    pd.DataFrame([proposal_row]).to_csv(working_path, index=False)
    freeze_proposal("ROUND_002", working_path, results_root=results_root)
    pd.DataFrame(
        [
            {
                **proposal_row,
                "replicate_id": "rep_001",
                "viability_percent": 72.5,
                "intact_patch_formation_pass": "yes",
            }
        ]
    ).to_csv(working_path, index=False)
    completed_bytes = working_path.read_bytes()

    formulations_path = tmp_path / "formulations.csv"
    pd.DataFrame(columns=FORMULATION_BASE_COLUMNS + registry.feature_names).to_csv(
        formulations_path,
        index=False,
    )
    observations_path = tmp_path / "observations.csv"
    pd.DataFrame(columns=OBSERVATION_COLUMNS).to_csv(observations_path, index=False)
    argv = [
        "run_round.py",
        str(working_path),
        "--formulations",
        str(formulations_path),
        "--observations",
        str(observations_path),
        "--output-dir",
        str(output_dir),
        "--total-candidate-pool",
        str(results_root / "total_candidate_pool.csv"),
    ]
    selector_calls: list[tuple[Path, list[str]]] = []
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        run_round_module,
        "_run",
        lambda script_path, extra_args: selector_calls.append((script_path, extra_args)),
    )

    def fail_reporting(*args: object, **kwargs: object) -> list[Path]:
        raise RuntimeError("simulated report failure")

    monkeypatch.setattr(
        run_round_module,
        "generate_completed_round_artifacts",
        fail_reporting,
    )
    with pytest.raises(RuntimeError, match="simulated report failure"):
        run_round_module.main()

    artifacts = round_artifact_paths("ROUND_002", results_root)
    assert artifacts.completed_csv.read_bytes() == completed_bytes
    observations_after_failure = pd.read_csv(observations_path)
    assert len(observations_after_failure) == 2
    assert selector_calls == []

    def succeed_reporting(
        formulations: pd.DataFrame,
        observations: pd.DataFrame,
        completed_candidates: pd.DataFrame,
        reports_dir: str | Path,
        batch_id: str,
    ) -> list[Path]:
        report_path = Path(reports_dir) / "report_summary.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(f"{batch_id} rerun report\n", encoding="utf-8")
        return [report_path]

    monkeypatch.setattr(
        run_round_module,
        "generate_completed_round_artifacts",
        succeed_reporting,
    )
    run_round_module.main()

    observations_after_retry = pd.read_csv(observations_path)
    assert len(observations_after_retry) == 2
    assert len(selector_calls) == 1
    assert (artifacts.reports_dir / "report_summary.txt").exists()
