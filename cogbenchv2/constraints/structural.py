"""Tier 1 Structural Constraints — deterministic, no ML model needed.

Level-specific rules that check keywords, word counts, concept counts, etc.
Every constraint here can be verified with pure string operations.
"""

import re
from collections import Counter

from cogbenchv2.constraints.base import Constraint, QuestionData, ConstraintResult
from cogbenchv2.config import (
    BLOOM_VERBS,
    R_MAX_CONCEPTS, R_MAX_ANSWER_WORDS, R_PASSAGE_OVERLAP,
    D_MAX_PASSAGE_OVERLAP,
    A_MIN_CONCEPTS,
    C_MIN_SPEC_MARKERS, C_MIN_ANSWER_WORDS,
)


# ─── Shared utilities ─────────────────────────────────────────────────────────

def _stem_words(words):
    """Simple suffix-stripping stemmer (no external deps)."""
    stemmed = []
    for w in words:
        w = w.lower()
        # Simple English stemming: remove common suffixes
        for suffix in ["ing", "tion", "sion", "ment", "ness", "ous", "ive",
                        "able", "ible", "ful", "less", "ly", "ed", "er", "est", "es", "s"]:
            if len(w) > len(suffix) + 2 and w.endswith(suffix):
                w = w[:-len(suffix)]
                break
        stemmed.append(w)
    return stemmed


def _word_overlap(words_a, words_b):
    """Fraction of words_a that appear in words_b (stemmed)."""
    if not words_a:
        return 0.0
    stems_a = set(_stem_words(words_a))
    stems_b = set(_stem_words(words_b))
    overlap = stems_a & stems_b
    return len(overlap) / len(stems_a)


def _contains_any(text_lower, patterns):
    """Check if text contains any of the patterns (case-insensitive)."""
    for p in patterns:
        if p in text_lower:
            return True, p
    return False, None


def _find_concepts_in_text(text_lower, concepts):
    """Find which key concepts appear in text (word-boundary matching)."""
    found = []
    for c in concepts:
        cl = c.lower()
        # Use word-boundary regex to avoid substring false positives
        # e.g., "cell" should NOT match in "cellular" or "cancel"
        pattern = r'\b' + re.escape(cl) + r'(?:s|es|ed|ing)?\b'
        if re.search(pattern, text_lower):
            found.append(c)
    return found


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — REMEMBER
# ═══════════════════════════════════════════════════════════════════════════════

class RememberVocabulary(Constraint):
    """R1: Uses Remember-level vocabulary or natural recall framing."""

    constraint_id = "R1"
    constraint_name = "remember_vocabulary"
    tier = "structural"

    # Natural interrogative patterns that signal factual recall
    RECALL_PATTERNS = [
        "what is", "what are", "what was", "what were", "what does", "what do",
        "what did", "what type", "what kind", "what form", "what part",
        "what vitamin", "what molecule", "what organ", "what structure",
        "what process", "what function", "what role", "what term",
        "what class", "what group", "what phylum", "what percentage",
        "what number", "what year", "what region", "what stage",
        "how many", "how much", "how long", "how old",
        "when did", "when does", "when was", "when is", "when were",
        "where is", "where are", "where does", "where do", "where did",
        "who is", "who was", "who are", "who did", "who discovered",
        "which is", "which are", "which was", "which of", "which type",
    ]

    def check(self, data: QuestionData) -> ConstraintResult:
        # Check canonical Bloom's verbs first
        found, verb = _contains_any(data.question_lower, BLOOM_VERBS[1])
        if found:
            return self._result(True, 1.0, f"Contains Remember verb: '{verb}'")

        # Check natural interrogative patterns
        found, pattern = _contains_any(data.question_lower, self.RECALL_PATTERNS)
        if found:
            return self._result(True, 0.9, f"Contains recall pattern: '{pattern}'")

        return self._result(False, 0.0,
                            f"No Remember verbs or recall patterns found. "
                            f"Expected Bloom's verbs {BLOOM_VERBS[1][:4]}... "
                            f"or patterns like 'What is', 'How many', 'When did'...")


class RememberSingleConcept(Constraint):
    """R2: Targets a single concept — at most 2 key concepts from passage appear."""

    constraint_id = "R2"
    constraint_name = "single_concept"
    tier = "structural"

    MAX_CONCEPTS = 2  # Allow up to 2 — some context is natural

    @staticmethod
    def _deduplicate_concepts(found):
        """Remove concepts that are substrings of other found concepts.

        E.g., if both 'cell' and 'cell cycle' are found, keep only 'cell cycle'.
        """
        deduped = []
        for c in found:
            # Skip if this concept is a substring of another found concept
            if any(c != other and c.lower() in other.lower() for other in found):
                continue
            deduped.append(c)
        return deduped

    def check(self, data: QuestionData) -> ConstraintResult:
        if not data.key_concepts:
            return self._result(True, 1.0, "No key concepts provided; skipped")

        found = _find_concepts_in_text(data.question_lower, data.key_concepts)
        found = self._deduplicate_concepts(found)
        n = len(found)

        if n <= self.MAX_CONCEPTS:
            return self._result(True, 1.0,
                                f"{n} concept(s) found: {found} (max {self.MAX_CONCEPTS})")
        return self._result(False, self.MAX_CONCEPTS / n,
                            f"{n} concepts found: {found} (max {self.MAX_CONCEPTS}). "
                            "Remember questions should target a single concept.")


class RememberShortAnswer(Constraint):
    """R3: Answer is short (factual) — <= 20 words."""

    constraint_id = "R3"
    constraint_name = "short_answer"
    tier = "structural"

    def check(self, data: QuestionData) -> ConstraintResult:
        n_words = len(data.answer_words)
        passed = n_words <= R_MAX_ANSWER_WORDS

        if passed:
            return self._result(True, 1.0,
                                f"Answer is {n_words} words (max {R_MAX_ANSWER_WORDS})")
        return self._result(False, R_MAX_ANSWER_WORDS / n_words,
                            f"Answer is {n_words} words (max {R_MAX_ANSWER_WORDS}). "
                            "Remember answers should be short, factual.")


class RememberExtractable(Constraint):
    """R4: Answer is findable in the passage — >=60% stemmed word overlap."""

    constraint_id = "R4"
    constraint_name = "answer_extractable"
    tier = "structural"

    def check(self, data: QuestionData) -> ConstraintResult:
        # Remove stop words from answer before checking overlap
        stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                "to", "and", "or", "for", "on", "at", "by", "with", "it"}
        answer_content = [w for w in data.answer_words if w not in stop]

        if not answer_content:
            return self._result(True, 1.0, "No content words in answer; skipped")

        overlap = _word_overlap(answer_content, data.passage_words)
        passed = overlap >= R_PASSAGE_OVERLAP

        if passed:
            return self._result(True, overlap,
                                f"{overlap:.0%} of answer words found in passage "
                                f"(threshold {R_PASSAGE_OVERLAP:.0%})")
        return self._result(False, overlap,
                            f"Only {overlap:.0%} of answer words found in passage "
                            f"(need >={R_PASSAGE_OVERLAP:.0%}). "
                            "Remember answers should be directly extractable.")


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — UNDERSTAND
# ═══════════════════════════════════════════════════════════════════════════════

class UnderstandVocabulary(Constraint):
    """D1: Uses Understand-level vocabulary or natural comprehension framing."""

    constraint_id = "D1"
    constraint_name = "understand_vocabulary"
    tier = "structural"

    # Natural patterns that signal comprehension-level thinking
    UNDERSTAND_PATTERNS = [
        "why is", "why are", "why does", "why do", "why did", "why was",
        "why were", "why would", "why can", "why might",
        "how does", "how do", "how is", "how are", "how did", "how was",
        "how can", "how would", "how might",
        "what is the difference", "what is the purpose", "what is the role",
        "what is the function", "what is the significance", "what is the meaning",
        "what is the relationship", "what is the importance",
        "what happens when", "what happens if", "what would happen",
        "what is the effect", "what is the impact", "what is the result",
        "in what way", "in your own words",
        "what does it mean", "what do you understand",
        "given that", "considering that",
    ]

    def check(self, data: QuestionData) -> ConstraintResult:
        # Check canonical Bloom's verbs first
        found, verb = _contains_any(data.question_lower, BLOOM_VERBS[2])
        if found:
            return self._result(True, 1.0, f"Contains Understand verb: '{verb}'")

        # Check natural comprehension patterns
        found, pattern = _contains_any(data.question_lower, self.UNDERSTAND_PATTERNS)
        if found:
            return self._result(True, 0.9, f"Contains comprehension pattern: '{pattern}'")

        return self._result(False, 0.0,
                            f"No Understand verbs or comprehension patterns found. "
                            f"Expected Bloom's verbs {BLOOM_VERBS[2][:4]}... "
                            f"or patterns like 'Why is', 'How does', 'What is the purpose'...")


class UnderstandNotCopied(Constraint):
    """D2: Answer is NOT copy-pasted from passage — <70% trigram overlap."""

    constraint_id = "D2"
    constraint_name = "answer_not_copied"
    tier = "structural"

    @staticmethod
    def _ngram_overlap(answer_words, passage_words, n=3):
        """Fraction of answer n-grams that appear verbatim in passage."""
        if len(answer_words) < n:
            return 0.0
        answer_ngrams = set(
            tuple(answer_words[i:i+n]) for i in range(len(answer_words) - n + 1)
        )
        if not answer_ngrams:
            return 0.0
        passage_ngrams = set(
            tuple(passage_words[i:i+n]) for i in range(len(passage_words) - n + 1)
        )
        return len(answer_ngrams & passage_ngrams) / len(answer_ngrams)

    def check(self, data: QuestionData) -> ConstraintResult:
        answer_words = data.answer_words
        passage_words = data.passage_words

        if len(answer_words) < 5:
            return self._result(True, 1.0, "Answer too short to evaluate; skipped")

        # 3-gram overlap detects verbatim copying better than single-word overlap
        trigram_overlap = self._ngram_overlap(answer_words, passage_words, n=3)
        passed = trigram_overlap < D_MAX_PASSAGE_OVERLAP

        if passed:
            return self._result(True, 1.0 - trigram_overlap,
                                f"{trigram_overlap:.0%} trigram overlap with passage "
                                f"(max {D_MAX_PASSAGE_OVERLAP:.0%}). Answer is paraphrased.")
        return self._result(False, 1.0 - trigram_overlap,
                            f"{trigram_overlap:.0%} trigram overlap with passage "
                            f"(max {D_MAX_PASSAGE_OVERLAP:.0%}). "
                            "Understand answers should paraphrase, not copy.")


class UnderstandMeaning(Constraint):
    """D4: Asks about meaning, not bare facts — contains how/why/explain/describe."""

    constraint_id = "D4"
    constraint_name = "asks_meaning"
    tier = "structural"

    MEANING_MARKERS = ["how", "why", "explain", "describe", "what does", "what do",
                       "in your own words", "in what way", "meaning of"]

    def check(self, data: QuestionData) -> ConstraintResult:
        found, marker = _contains_any(data.question_lower, self.MEANING_MARKERS)
        if found:
            return self._result(True, 1.0, f"Contains meaning marker: '{marker}'")
        return self._result(False, 0.0,
                            "No meaning markers found (how/why/explain/describe). "
                            "Understand questions should ask about meaning.")


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 3 — APPLY
# ═══════════════════════════════════════════════════════════════════════════════

class ApplyVocabulary(Constraint):
    """P1: Uses Apply-level vocabulary or scenario-based application framing."""

    constraint_id = "P1"
    constraint_name = "apply_vocabulary"
    tier = "structural"

    # Natural patterns that signal application-level thinking
    APPLY_PATTERNS = [
        # Scenario framing (if/given/suppose/consider + problem)
        "if a ", "if an ", "if the ", "if you ", "if we ",
        "given that", "given a ", "given an ", "given the ",
        "suppose ", "assuming ", "consider a ", "consider the ",
        "imagine ", "say that ",
        # Problem-solving cues
        "how would you", "how could you", "how can you",
        "what would happen if", "what would be the result",
        "what steps", "what procedure", "what method",
        "a researcher", "a student", "a scientist", "an engineer",
        "a patient", "a doctor",
        # Computation/procedure cues
        "find the", "predict the", "estimate the",
        "what is the value", "what would the",
    ]

    def check(self, data: QuestionData) -> ConstraintResult:
        # Check canonical Bloom's verbs first
        found, verb = _contains_any(data.question_lower, BLOOM_VERBS[3])
        if found:
            return self._result(True, 1.0, f"Contains Apply verb: '{verb}'")

        # Check scenario-based application patterns
        found, pattern = _contains_any(data.question_lower, self.APPLY_PATTERNS)
        if found:
            return self._result(True, 0.9, f"Contains application pattern: '{pattern}'")

        return self._result(False, 0.0,
                            f"No Apply verbs or application patterns found. "
                            f"Expected Bloom's verbs {BLOOM_VERBS[3][:4]}... "
                            f"or patterns like 'If a...', 'Given that...', 'How would you...'...")


class ApplyMethodReference(Constraint):
    """P3: References a concept, method, or principle from the passage.

    Apply questions should ground their scenario in passage content.
    Checks both question and answer, using key_concepts as the primary
    source (methods_principles is often poorly extracted).
    """

    constraint_id = "P3"
    constraint_name = "method_reference"
    tier = "structural"

    def check(self, data: QuestionData) -> ConstraintResult:
        combined_lower = data.question_lower + " " + data.answer_lower

        # Use key_concepts as primary check (more reliable than methods_principles)
        if data.key_concepts:
            found = _find_concepts_in_text(combined_lower, data.key_concepts)
            if found:
                return self._result(True, 1.0,
                                    f"References passage concept: {found[:3]}")

        # Fallback to methods_principles if available
        if data.methods_principles:
            # Filter out garbage entries (too short or too generic)
            good_methods = [m for m in data.methods_principles
                           if len(m.split()) <= 5 and len(m) > 3]
            if good_methods:
                found = _find_concepts_in_text(combined_lower, good_methods)
                if found:
                    return self._result(True, 1.0,
                                        f"References method/principle: {found[0]}")

        # Last resort: check stemmed word overlap between Q+A and passage
        if data.passage_words:
            combined_words = re.findall(r'[a-z]+', combined_lower)
            overlap = _word_overlap(combined_words, data.passage_words)
            if overlap >= 0.20:
                return self._result(True, 0.8,
                                    f"Q+A shares {overlap:.0%} vocabulary with passage")

        if not data.key_concepts and not data.methods_principles:
            return self._result(True, 1.0, "No concepts/methods provided; skipped")

        return self._result(False, 0.0,
                            "Q+A does not reference passage concepts or methods.")


class ApplySpecificResult(Constraint):
    """P4: Answer contains a specific result — number, named outcome, or step."""

    constraint_id = "P4"
    constraint_name = "specific_result"
    tier = "structural"

    # Patterns that indicate a specific result or concrete reasoning
    RESULT_PATTERNS = [
        r'\d+',           # Any number
        r'step\s*\d',     # "step 1", "step 2"
        r'first|second|third|finally',  # Procedural steps
        r'result|outcome|solution|output',  # Result words
        r'therefore|thus|hence|consequently',  # Conclusion markers
        r'equals?|yields?|produces?|gives?',   # Computation results
        r'because|since|due to|as a result',   # Causal reasoning
        r'would\s+(be|cause|lead|result|produce|increase|decrease)',  # Predicted outcomes
        r'this\s+(means|indicates|suggests|shows|leads|causes|results)',  # Inference
        r'the\s+(answer|solution|result|value|effect|consequence)\s+is',  # Direct answers
        r'in\s+this\s+(case|scenario|situation)',  # Scenario application
        r'by\s+(using|applying|following|performing)',  # Method application
    ]

    def check(self, data: QuestionData) -> ConstraintResult:
        answer = data.answer_lower
        for pattern in self.RESULT_PATTERNS:
            if re.search(pattern, answer):
                match = re.search(pattern, answer).group()
                return self._result(True, 1.0,
                                    f"Answer contains specific result: '{match}'")

        return self._result(False, 0.0,
                            "Answer lacks specific result (no numbers, steps, or outcomes). "
                            "Apply answers should contain concrete results.")


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 4 — ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyzeVocabulary(Constraint):
    """A1: Uses Analyze-level vocabulary or natural analytical framing."""

    constraint_id = "A1"
    constraint_name = "analyze_vocabulary"
    tier = "structural"

    # Natural patterns that signal analysis-level thinking
    ANALYZE_PATTERNS = [
        "how do", "how does", "how is", "how are",
        "what is the relationship", "what is the connection",
        "what is the difference", "what are the differences",
        "what is the similarity", "what are the similarities",
        "in what way", "in what ways",
        "what role does", "what role do",
        "how do they differ", "how does it differ",
        "differ from", "different from", "differs from",
        "similar to", "compared to", "in comparison",
        "what factors", "what causes", "what contributes",
        "why does", "why do", "why is", "why are",
        "the relationship between", "the connection between",
        "the difference between", "the similarities between",
    ]

    def check(self, data: QuestionData) -> ConstraintResult:
        # Check canonical Bloom's verbs first
        found, verb = _contains_any(data.question_lower, BLOOM_VERBS[4])
        if found:
            return self._result(True, 1.0, f"Contains Analyze verb: '{verb}'")

        # Check natural analytical patterns
        found, pattern = _contains_any(data.question_lower, self.ANALYZE_PATTERNS)
        if found:
            return self._result(True, 0.9, f"Contains analytical pattern: '{pattern}'")

        return self._result(False, 0.0,
                            f"No Analyze verbs or analytical patterns found. "
                            f"Expected Bloom's verbs {BLOOM_VERBS[4][:4]}... "
                            f"or patterns like 'How do X differ', 'What is the relationship'...")


class AnalyzeMultipleConcepts(Constraint):
    """A2: References >=2 concepts from the passage."""

    constraint_id = "A2"
    constraint_name = "multiple_concepts"
    tier = "structural"

    # Common acronym expansions in science textbooks
    ACRONYM_MAP = {
        "rer": "rough endoplasmic reticulum",
        "ser": "smooth endoplasmic reticulum",
        "er": "endoplasmic reticulum",
        "dna": "deoxyribonucleic acid",
        "rna": "ribonucleic acid",
        "mrna": "messenger rna",
        "trna": "transfer rna",
        "atp": "adenosine triphosphate",
        "adp": "adenosine diphosphate",
        "nadh": "nicotinamide adenine dinucleotide",
        "gdp": "gross domestic product",
        "cns": "central nervous system",
        "pns": "peripheral nervous system",
    }

    def _find_with_acronyms(self, text_lower, concepts):
        """Find concepts in text, also matching acronyms."""
        found = _find_concepts_in_text(text_lower, concepts)

        # Also check if text contains acronyms that match concept expansions
        for acronym, expansion in self.ACRONYM_MAP.items():
            if re.search(r'\b' + re.escape(acronym) + r'\b', text_lower):
                for c in concepts:
                    if c.lower() not in [f.lower() for f in found]:
                        if (acronym in c.lower() or c.lower() in expansion
                                or expansion in c.lower()):
                            found.append(c)

        return found

    def check(self, data: QuestionData) -> ConstraintResult:
        if not data.key_concepts:
            return self._result(True, 1.0, "No key concepts provided; skipped")

        # Check question + answer combined (analysis often names concepts in answer)
        combined_lower = data.question_lower + " " + data.answer_lower
        found = self._find_with_acronyms(combined_lower, data.key_concepts)

        # Deduplicate substring matches
        deduped = []
        for c in found:
            if not any(c != other and c.lower() in other.lower() for other in found):
                deduped.append(c)
        found = deduped if deduped else found

        n = len(found)
        passed = n >= A_MIN_CONCEPTS
        score = min(n / A_MIN_CONCEPTS, 1.0)

        if passed:
            return self._result(True, score,
                                f"{n} concepts found: {found[:4]} (min {A_MIN_CONCEPTS})")
        return self._result(False, score,
                            f"Only {n} concept(s) found: {found} (need >={A_MIN_CONCEPTS}). "
                            "Analyze questions must connect multiple ideas.")


class AnalyzeRelationship(Constraint):
    """A3: Asks about relationships between concepts."""

    constraint_id = "A3"
    constraint_name = "asks_relationship"
    tier = "structural"

    RELATIONSHIP_MARKERS = [
        "between", "differ", "relate", "compare", "versus", "whereas",
        "how does", "affect", "influence", "impact", "connection",
        "relationship", "similarity", "difference", "in contrast",
        "as opposed to", "while", "on the other hand",
    ]

    def check(self, data: QuestionData) -> ConstraintResult:
        found, marker = _contains_any(data.question_lower, self.RELATIONSHIP_MARKERS)
        if found:
            return self._result(True, 1.0, f"Contains relationship marker: '{marker}'")
        return self._result(False, 0.0,
                            "No relationship markers found (between/differ/compare/affect). "
                            "Analyze questions should ask about relationships.")


class AnalyzeAnswerCoverage(Constraint):
    """A4: Answer addresses all referenced concepts from the question."""

    constraint_id = "A4"
    constraint_name = "answer_coverage"
    tier = "structural"

    def check(self, data: QuestionData) -> ConstraintResult:
        if not data.key_concepts:
            return self._result(True, 1.0, "No key concepts provided; skipped")

        # Find concepts in question
        q_concepts = _find_concepts_in_text(data.question_lower, data.key_concepts)
        if not q_concepts:
            return self._result(True, 1.0, "No passage concepts in question; skipped")

        # Check which are also in the answer
        a_concepts = _find_concepts_in_text(data.answer_lower, q_concepts)
        missing = [c for c in q_concepts if c not in a_concepts]
        coverage = len(a_concepts) / len(q_concepts)

        passed = len(missing) == 0
        if passed:
            return self._result(True, coverage,
                                f"Answer covers all {len(q_concepts)} concepts: {q_concepts}")
        return self._result(False, coverage,
                            f"Answer missing concepts: {missing}. "
                            f"Only covers {len(a_concepts)}/{len(q_concepts)}.")


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 5 — EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluateVocabulary(Constraint):
    """E1: Uses Evaluate-level vocabulary."""

    constraint_id = "E1"
    constraint_name = "evaluate_vocabulary"
    tier = "structural"

    def check(self, data: QuestionData) -> ConstraintResult:
        found, verb = _contains_any(data.question_lower, BLOOM_VERBS[5])
        if found:
            return self._result(True, 1.0, f"Contains Evaluate verb: '{verb}'")
        return self._result(False, 0.0,
                            f"No Evaluate verbs found. Expected one of: {BLOOM_VERBS[5][:6]}...")


class EvaluateClaim(Constraint):
    """E2: Presents something to evaluate — a claim, position, or judgment prompt."""

    constraint_id = "E2"
    constraint_name = "presents_claim"
    tier = "structural"

    CLAIM_MARKERS = [
        # Explicit claim/stance prompts
        "do you agree", "to what extent", "is it justified", "is it true",
        "is it valid", "would you recommend", "is it appropriate",
        "argue for or against", "defend or refute", "is it reasonable",
        "critique", "evaluate the claim", "evaluate this claim",
        "assess whether", "some argue", "the claim",
        "it has been claimed", "one could argue", "is it fair to say",
        "claims that", "it is claimed", "the argument",
        # Should/could/would judgment prompts
        "should ", "should be", "should not",
        "could be considered", "can be considered",
        "would it be", "is it possible that",
        "is it accurate", "is it correct", "is it fair",
        # Justify/defend prompts
        "justify your", "justify this", "provide reasons",
        "give reasons", "support your",
        "provide evidence", "provide your reasoning",
        # Broader evaluation phrasings
        "strengths and weaknesses", "strengths and limitations",
        "advantages and disadvantages", "pros and cons",
        "how effective", "how appropriate", "how adequate",
        "how justified", "how significant", "how valid",
        "how important", "how well does", "how successful",
        "is there sufficient", "is there enough evidence",
        "what are the limitations", "what are the drawbacks",
        "what are the implications", "what are the consequences",
        "evaluate whether", "evaluate how", "evaluate the",
        "critically assess", "critically evaluate", "critically analyze",
        "weigh the", "judge the", "merit of",
        # Position-taking patterns
        "your position", "your stance", "your opinion",
        "your argument", "your view",
        "take a position", "take a stance",
    ]

    def check(self, data: QuestionData) -> ConstraintResult:
        found, marker = _contains_any(data.question_lower, self.CLAIM_MARKERS)
        if found:
            return self._result(True, 1.0, f"Contains claim/judgment prompt: '{marker}'")
        return self._result(False, 0.0,
                            "No claim/judgment markers found. "
                            "Evaluate questions should present something to evaluate.")


class EvaluateEvidenceRequest(Constraint):
    """E3: Asks for evidence-based reasoning."""

    constraint_id = "E3"
    constraint_name = "evidence_request"
    tier = "structural"

    EVIDENCE_MARKERS = [
        "based on", "evidence", "criteria", "justify", "support",
        "why or why not", "provide reasons", "give evidence",
        "what evidence", "using examples", "with reference to",
    ]

    def check(self, data: QuestionData) -> ConstraintResult:
        found, marker = _contains_any(data.question_lower, self.EVIDENCE_MARKERS)
        if found:
            return self._result(True, 1.0, f"Contains evidence request: '{marker}'")
        return self._result(False, 0.0,
                            "No evidence markers found (based on/evidence/justify). "
                            "Evaluate questions should ask for evidence-based reasoning.")


class EvaluateArgumentation(Constraint):
    """E4: Answer contains argumentation — has contrastive markers."""

    constraint_id = "E4"
    constraint_name = "answer_argumentation"
    tier = "structural"

    CONTRASTIVE_MARKERS = [
        # Classic contrastive markers
        "however", "although", "on the other hand", "nevertheless",
        "conversely", "in contrast", "despite", "whereas",
        "notwithstanding", "on the contrary", "nonetheless",
        "but ", "yet ", "while ",
        # Argumentation markers (supporting + concluding)
        "therefore", "thus", "hence", "consequently",
        "because", "since ", "as a result",
        "this suggests", "this indicates", "this shows",
        "the evidence", "the data",
        # Evaluative language
        "more effective", "less effective", "better", "worse",
        "stronger", "weaker", "superior", "inferior",
        "significant", "insufficient", "adequate", "inadequate",
        "valid", "invalid", "justified", "unjustified",
        # Reasoning structure
        "on one hand", "first,", "second,", "finally,",
        "in conclusion", "overall", "in summary",
        "the strength", "the weakness", "the limitation",
        "the advantage", "the disadvantage",
    ]

    def check(self, data: QuestionData) -> ConstraintResult:
        found, marker = _contains_any(data.answer_lower, self.CONTRASTIVE_MARKERS)
        if found:
            return self._result(True, 1.0,
                                f"Answer contains contrastive marker: '{marker}'")
        return self._result(False, 0.0,
                            "Answer lacks contrastive markers (however/although/but). "
                            "Evaluate answers should contain argumentation.")


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 6 — CREATE
# ═══════════════════════════════════════════════════════════════════════════════

class CreateVocabulary(Constraint):
    """C1: Uses Create-level vocabulary."""

    constraint_id = "C1"
    constraint_name = "create_vocabulary"
    tier = "structural"

    def check(self, data: QuestionData) -> ConstraintResult:
        found, verb = _contains_any(data.question_lower, BLOOM_VERBS[6])
        if found:
            return self._result(True, 1.0, f"Contains Create verb: '{verb}'")
        return self._result(False, 0.0,
                            f"No Create verbs found. Expected one of: {BLOOM_VERBS[6][:6]}...")


class CreateSpecifications(Constraint):
    """C3: Specifies requirements — >=2 specification markers."""

    constraint_id = "C3"
    constraint_name = "specifications"
    tier = "structural"

    SPEC_MARKERS = [
        "must", "should", "include", "at least", "using",
        "ensure", "incorporate", "consider", "address", "that includes",
        "consisting of", "requirements", "criteria",
    ]

    def check(self, data: QuestionData) -> ConstraintResult:
        q = data.question_lower
        found = [m for m in self.SPEC_MARKERS if m in q]
        n = len(found)

        passed = n >= C_MIN_SPEC_MARKERS
        score = min(n / C_MIN_SPEC_MARKERS, 1.0)

        if passed:
            return self._result(True, score,
                                f"{n} specification markers found: {found[:4]} "
                                f"(min {C_MIN_SPEC_MARKERS})")
        return self._result(False, score,
                            f"Only {n} spec marker(s): {found} (need >={C_MIN_SPEC_MARKERS}). "
                            "Create questions should specify what to produce.")


class CreateSubstantialAnswer(Constraint):
    """C4: Answer is substantial and open-ended — >50 words."""

    constraint_id = "C4"
    constraint_name = "substantial_answer"
    tier = "structural"

    def check(self, data: QuestionData) -> ConstraintResult:
        n_words = len(data.answer_words)
        passed = n_words > C_MIN_ANSWER_WORDS
        score = min(n_words / C_MIN_ANSWER_WORDS, 1.0)

        if passed:
            return self._result(True, score,
                                f"Answer is {n_words} words (min {C_MIN_ANSWER_WORDS})")
        return self._result(False, score,
                            f"Answer is only {n_words} words (need >{C_MIN_ANSWER_WORDS}). "
                            "Create answers should be substantial.")


# ─── Constraint lists per level ───────────────────────────────────────────────

REMEMBER_CONSTRAINTS = [
    RememberVocabulary(),
    RememberSingleConcept(),
    RememberShortAnswer(),
    RememberExtractable(),
]

UNDERSTAND_CONSTRAINTS = [
    UnderstandVocabulary(),
    UnderstandNotCopied(),
    # D3 is in semantic.py (NLI-based)
    UnderstandMeaning(),
]

APPLY_CONSTRAINTS = [
    ApplyVocabulary(),
    # P2 is in semantic.py (NLI-based)
    ApplyMethodReference(),
    ApplySpecificResult(),
]

ANALYZE_CONSTRAINTS = [
    AnalyzeVocabulary(),
    AnalyzeMultipleConcepts(),
    AnalyzeRelationship(),
    AnalyzeAnswerCoverage(),
]

EVALUATE_CONSTRAINTS = [
    EvaluateVocabulary(),
    EvaluateClaim(),
    EvaluateEvidenceRequest(),
    EvaluateArgumentation(),
]

CREATE_CONSTRAINTS = [
    CreateVocabulary(),
    # C2 is in semantic.py (NLI-based)
    CreateSpecifications(),
    CreateSubstantialAnswer(),
]
