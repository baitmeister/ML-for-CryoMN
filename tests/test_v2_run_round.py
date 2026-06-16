from __future__ import annotations

from pathlib import Path

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
