from __future__ import annotations

from pathlib import Path

from run_round import NEXT_ROUND_SUMMARY_PATH, _resolve_current_summary_path


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
