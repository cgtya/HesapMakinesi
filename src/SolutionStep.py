from dataclasses import dataclass, field
from typing import List

@dataclass
class SolutionStep:
    input_expr: str       # latex
    output_expr: str      # latex
    rule_name: str        # kural adı
    description: str      # açıklama
    substeps: List['SolutionStep'] = field(default_factory=list)

    def to_json(self):
        return {
            "input": self.input_expr,
            "output": self.output_expr,
            "rule": self.rule_name,
            "description": self.description,
            "substeps": [step.to_json() for step in self.substeps]
        }