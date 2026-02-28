"""Universal constraints — must be satisfied by ALL Bloom's levels.

U1: Output is actually a question (ends with "?")
U2: Question is 10-150 words
U3: Question is about the passage topic (>=3 key terms present)
U4: No degenerate output (no excessive repetition, not empty)
"""

import re
from collections import Counter

from cogbenchv2.constraints.base import Constraint, QuestionData, ConstraintResult
from cogbenchv2.config import U_MIN_WORDS, U_MAX_WORDS, U_MIN_PASSAGE_TERMS, U_MAX_WORD_REPEAT


class IsQuestion(Constraint):
    """U1: Output is a question or task prompt — ends with '?' or contains interrogative structure."""

    constraint_id = "U1"
    constraint_name = "is_question"
    tier = "universal"

    # Task-style openings common at higher Bloom's levels (L5-L6)
    TASK_STARTERS = [
        "evaluate", "assess", "critique", "judge", "argue", "defend",
        "design", "develop", "propose", "construct", "formulate", "create",
        "devise", "plan", "compose", "synthesize",
    ]

    # Endings that indicate a response is requested (L5 evaluation prompts)
    RESPONSE_ENDINGS = [
        "your argument", "your position", "your response",
        "your reasoning", "your answer", "your analysis",
        "your view", "your opinion", "your stance",
        "the passage", "from the passage", "based on the passage",
        "with evidence", "with examples",
    ]

    def check(self, data: QuestionData) -> ConstraintResult:
        text = data.question.strip() if data.question else ""
        if not text:
            return self._result(False, 0.0, "Empty question")

        # Primary check: ends with "?"
        if text.endswith("?"):
            return self._result(True, 1.0, "Ends with '?'")

        text_lower = text.lower()

        # Secondary: task-style prompt (common for Evaluate/Create levels)
        for starter in self.TASK_STARTERS:
            if text_lower.startswith(starter):
                return self._result(True, 0.8,
                                    f"Task prompt starting with '{starter}'")

        # Tertiary: ends with a response-request pattern (L5 style)
        for ending in self.RESPONSE_ENDINGS:
            if text_lower.rstrip(".").endswith(ending):
                return self._result(True, 0.7,
                                    f"Response-request ending: '{ending}'")

        # Quaternary: contains a question word
        question_words = ["how", "why", "what", "which", "when", "where", "who",
                          "can", "could", "would", "should", "does", "do", "is"]
        for qw in question_words:
            if re.search(rf'\b{qw}\b', text_lower):
                return self._result(True, 0.6,
                                    f"Contains question word '{qw}'")

        return self._result(False, 0.0,
                            f"Ends with '{text[-1]}', not '?', and no task/question structure found")


class WordCount(Constraint):
    """U2: Question is 10-150 words."""

    constraint_id = "U2"
    constraint_name = "word_count"
    tier = "universal"

    def check(self, data: QuestionData) -> ConstraintResult:
        n_words = len(data.question_words)

        # Remember questions are naturally shorter (factual recall)
        min_words = 5 if data.target_level == 1 else U_MIN_WORDS

        if n_words < min_words:
            return self._result(False, n_words / min_words,
                                f"Too short: {n_words} words (min {min_words})")
        if n_words > U_MAX_WORDS:
            return self._result(False, U_MAX_WORDS / n_words,
                                f"Too long: {n_words} words (max {U_MAX_WORDS})")

        return self._result(True, 1.0, f"{n_words} words (valid range)")


class PassageRelevance(Constraint):
    """U3: Output is about the passage topic — >=2 key terms appear in question+answer combined.

    Checks both question and answer because some levels (e.g., Remember) produce
    narrow questions that reference only one concept, with the answer providing
    additional passage context. Together they prove passage relevance.
    """

    constraint_id = "U3"
    constraint_name = "passage_relevance"
    tier = "universal"

    @staticmethod
    def _tokenize(text):
        """Split text into lowercase word tokens."""
        return re.findall(r'[a-z]+', text.lower())

    @staticmethod
    def _stem(word):
        """Simple suffix-stripping stemmer."""
        for suffix in ["ing", "tion", "sion", "ment", "ness", "ous", "ive",
                        "able", "ible", "ful", "less", "ly", "ed", "er", "est", "es", "s"]:
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                return word[:-len(suffix)]
        return word

    def _concept_match(self, concept, text_lower, text_stems):
        """Check if concept matches in text via word-boundary regex or stem overlap."""
        c_lower = concept.lower()
        # Word-boundary match (handles plurals: cell matches cells)
        pattern = r'\b' + re.escape(c_lower) + r'(?:s|es|ed|ing)?\b'
        if re.search(pattern, text_lower):
            return True
        # Stem-based: if all stems of the concept appear in text stems
        c_tokens = self._tokenize(c_lower)
        if c_tokens:
            c_stems = {self._stem(w) for w in c_tokens}
            return c_stems.issubset(text_stems)
        return False

    def check(self, data: QuestionData) -> ConstraintResult:
        if not data.key_concepts:
            return self._result(True, 1.0, "No key concepts provided; skipped")

        # Check question + answer combined for passage relevance
        combined_lower = data.question_lower + " " + data.answer_lower
        combined_tokens = self._tokenize(combined_lower)
        combined_stems = {self._stem(w) for w in combined_tokens}

        found = [c for c in data.key_concepts
                 if self._concept_match(c, combined_lower, combined_stems)]
        n_found = len(found)
        total = len(data.key_concepts)

        # Level 1 (Remember) targets a single concept, so 1 key term is
        # sufficient to prove passage relevance.  For all other levels the
        # standard threshold (>=2) applies.
        min_terms = 1 if data.target_level == 1 else U_MIN_PASSAGE_TERMS

        passed = n_found >= min_terms
        score = min(n_found / min_terms, 1.0)

        if passed:
            details = f"{n_found}/{total} key terms in Q+A: {found[:5]}"
        else:
            missing = [c for c in data.key_concepts
                       if not self._concept_match(c, combined_lower, combined_stems)]
            details = (f"Only {n_found}/{total} key terms in Q+A (need >={min_terms}). "
                       f"Found: {found}, missing: {missing[:3]}")

        return self._result(passed, score, details)


class NoDegenerateOutput(Constraint):
    """U4: No degenerate output — no word repeated >3 times, not empty."""

    constraint_id = "U4"
    constraint_name = "no_degenerate"
    tier = "universal"

    def check(self, data: QuestionData) -> ConstraintResult:
        text = data.question.strip() if data.question else ""

        if not text:
            return self._result(False, 0.0, "Empty output")

        if len(text) < 5:
            return self._result(False, 0.0, f"Output too short: '{text}'")

        # Check for excessive word repetition (skip common stop words)
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                      "to", "and", "or", "for", "on", "at", "by", "with", "that",
                      "this", "it", "from", "as", "be", "has", "have", "had",
                      "not", "but", "what", "how", "do", "does", "did", "can",
                      "if", "which", "their", "its", "they", "you", "your"}

        words = data.question_words
        content_words = [w for w in words if w not in stop_words]
        if content_words:
            counts = Counter(content_words)
            most_common_word, most_common_count = counts.most_common(1)[0]
            if most_common_count > U_MAX_WORD_REPEAT:
                return self._result(False, U_MAX_WORD_REPEAT / most_common_count,
                                    f"Word '{most_common_word}' repeated {most_common_count}x "
                                    f"(max {U_MAX_WORD_REPEAT})")

        return self._result(True, 1.0, "No degenerate patterns detected")


# All universal constraints
UNIVERSAL_CONSTRAINTS = [
    IsQuestion(),
    WordCount(),
    PassageRelevance(),
    NoDegenerateOutput(),
]
