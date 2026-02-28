"""Adversarial mode constraints.

In adversarial mode, the model must:
1. Satisfy all STRUCTURAL constraints for the TARGET level (except vocabulary)
2. Use VOCABULARY from a DIFFERENT level

This tests whether the model understands cognitive complexity beyond just
using the right keywords.
"""

from cogbenchv2.constraints.base import Constraint, QuestionData, ConstraintResult
from cogbenchv2.config import BLOOM_VERBS, ADVERSARIAL_PAIRINGS


def _contains_any(text_lower, patterns):
    """Check if text contains any of the patterns."""
    for p in patterns:
        if p in text_lower:
            return True, p
    return False, None


class AdversarialVocabulary(Constraint):
    """AV: In adversarial mode, uses vocabulary from the WRONG level (as specified).

    For example, if generating an Analyze question using Remember verbs,
    this checks that Remember verbs appear in the question.
    """

    constraint_id = "AV"
    constraint_name = "adversarial_vocabulary"
    tier = "adversarial"

    def check(self, data: QuestionData) -> ConstraintResult:
        if data.mode != "adversarial" or data.vocab_level is None:
            return self._result(True, 1.0, "Not adversarial mode; skipped")

        vocab_verbs = BLOOM_VERBS.get(data.vocab_level, [])
        if not vocab_verbs:
            return self._result(True, 1.0,
                                f"No verbs defined for level {data.vocab_level}; skipped")

        found, verb = _contains_any(data.question_lower, vocab_verbs)
        level_name = {1: "Remember", 2: "Understand", 3: "Apply",
                      4: "Analyze", 5: "Evaluate", 6: "Create"}.get(data.vocab_level, "?")

        if found:
            return self._result(True, 1.0,
                                f"Uses vocabulary from L{data.vocab_level} ({level_name}): '{verb}'")
        return self._result(False, 0.0,
                            f"Does not use vocabulary from L{data.vocab_level} ({level_name}). "
                            f"Expected one of: {vocab_verbs[:5]}...")


class AdversarialNoTargetVocab(Constraint):
    """AN: In adversarial mode, should NOT use vocabulary from the TARGET level.

    This is a soft constraint — it's harder but more meaningful when the model
    avoids the "easy" verbs for its target level.
    """

    constraint_id = "AN"
    constraint_name = "no_target_vocabulary"
    tier = "adversarial"

    def check(self, data: QuestionData) -> ConstraintResult:
        if data.mode != "adversarial":
            return self._result(True, 1.0, "Not adversarial mode; skipped")

        target_verbs = BLOOM_VERBS.get(data.target_level, [])
        if not target_verbs:
            return self._result(True, 1.0,
                                f"No verbs defined for target level {data.target_level}; skipped")

        found, verb = _contains_any(data.question_lower, target_verbs)
        level_name = {1: "Remember", 2: "Understand", 3: "Apply",
                      4: "Analyze", 5: "Evaluate", 6: "Create"}.get(data.target_level, "?")

        if not found:
            return self._result(True, 1.0,
                                f"Avoids target L{data.target_level} ({level_name}) vocabulary")
        return self._result(False, 0.5,
                            f"Uses target vocabulary '{verb}' from L{data.target_level} "
                            f"({level_name}). Adversarial mode should use only the "
                            f"specified vocabulary level.")


ADVERSARIAL_CONSTRAINTS = [
    AdversarialVocabulary(),
    AdversarialNoTargetVocab(),
]
