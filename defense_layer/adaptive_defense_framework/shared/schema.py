from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class DialogueSample:
    sample_id: str
    source: str
    label: Optional[int]
    turns: List[Turn]
    meta: Dict[str, Any] = field(default_factory=dict)
