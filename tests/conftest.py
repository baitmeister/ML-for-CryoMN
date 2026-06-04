from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_SRC = PROJECT_ROOT / "src" / "08_multi_objective"
V2_RECORD = V2_SRC / "03_record_results"
V2_VISUALIZATION = V2_SRC / "04_visualization"
for path in [V2_SRC, V2_RECORD, V2_VISUALIZATION]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
