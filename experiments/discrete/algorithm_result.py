from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class AlgorithmResult:
    algorithm_name: str
    best_tour: List[int]
    best_cost: float
    cost_history: List[float]
    elapsed_time: float
    iterations: int
    parameters: Dict[str, Any]
    memory_usage: float = 0.0
