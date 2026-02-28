"""Base classes for constraint checking."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConstraintResult:
    """Result of checking a single constraint on a question."""

    constraint_id: str       # e.g. "U1", "R3", "D2"
    constraint_name: str     # Human-readable name
    passed: bool             # Did the constraint pass?
    score: float             # 0.0-1.0 continuous score (for loose matching)
    details: str             # Explanation of why it passed/failed
    tier: str                # "universal", "structural", "semantic", "adversarial"

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.constraint_id} ({self.constraint_name}): {self.details}"


@dataclass
class QuestionData:
    """All data needed to evaluate a generated question.

    This is what gets passed to every constraint checker.
    """

    question: str                      # The generated question text
    answer: str                        # The reference answer
    passage: str                       # The source passage text
    target_level: int                  # Target Bloom's level (1-6)
    key_concepts: list = field(default_factory=list)     # Key concepts from passage
    methods_principles: list = field(default_factory=list)  # Methods/principles from passage
    mode: str = "standard"             # "standard" or "adversarial"
    vocab_level: Optional[int] = None  # For adversarial: which level's vocabulary to use
    subject: str = ""
    passage_id: str = ""

    @property
    def question_lower(self) -> str:
        return self.question.lower() if self.question else ""

    @property
    def answer_lower(self) -> str:
        return self.answer.lower() if self.answer else ""

    @property
    def passage_lower(self) -> str:
        return self.passage.lower() if self.passage else ""

    @property
    def question_words(self) -> list:
        """Question split into words (lowercase, stripped of punctuation)."""
        import re
        return re.findall(r'\b\w+\b', self.question_lower)

    @property
    def answer_words(self) -> list:
        """Answer split into words (lowercase, stripped of punctuation)."""
        import re
        return re.findall(r'\b\w+\b', self.answer_lower)

    @property
    def passage_words(self) -> list:
        """Passage split into words (lowercase, stripped of punctuation)."""
        import re
        return re.findall(r'\b\w+\b', self.passage_lower)


class Constraint:
    """Base class for all constraints.

    Subclasses must implement check(data: QuestionData) -> ConstraintResult.
    """

    constraint_id: str = ""
    constraint_name: str = ""
    tier: str = ""  # "universal", "structural", "semantic", "adversarial"

    def check(self, data: QuestionData) -> ConstraintResult:
        raise NotImplementedError

    def _result(self, passed: bool, score: float, details: str) -> ConstraintResult:
        """Helper to build a ConstraintResult with this constraint's metadata."""
        return ConstraintResult(
            constraint_id=self.constraint_id,
            constraint_name=self.constraint_name,
            passed=passed,
            score=score,
            details=details,
            tier=self.tier,
        )
