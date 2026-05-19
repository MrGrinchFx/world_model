"""Student-side scoreboard metric helpers.

Official grading uses the locked implementation in `wm_hw.official_metrics`.
Students may use this file for experiments and report diagnostics.
"""

from __future__ import annotations

from wm_hw.official_metrics import compute_official_metrics as compute_scoreboard_metrics


__all__ = ["compute_scoreboard_metrics"]
