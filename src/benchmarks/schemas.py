from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkCase:
    case_id: str
    base_id: str
    task_id: str
    task_type: str
    modality: str
    chart_type: str
    states: List[Dict[str, float]]
    question: str
    gold_answer: Any
    answer_type: str
    target_variable: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Connects a positive and negative assertion pair for Acc++ evaluation.
    # Both cases in a pair share the same pair_id; open-ended cases have None.
    pair_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)