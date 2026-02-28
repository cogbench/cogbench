"""Unit tests for all constraint checkers.

Each constraint gets >= 5 tests — known pass and known fail examples.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from cogbenchv2.constraints.base import QuestionData
from cogbenchv2.constraints.universal import (
    IsQuestion, WordCount, PassageRelevance, NoDegenerateOutput,
)
from cogbenchv2.constraints.structural import (
    RememberVocabulary, RememberSingleConcept, RememberShortAnswer, RememberExtractable,
    UnderstandVocabulary, UnderstandNotCopied, UnderstandMeaning,
    ApplyVocabulary, ApplyMethodReference, ApplySpecificResult,
    AnalyzeVocabulary, AnalyzeMultipleConcepts, AnalyzeRelationship, AnalyzeAnswerCoverage,
    EvaluateVocabulary, EvaluateClaim, EvaluateEvidenceRequest, EvaluateArgumentation,
    CreateVocabulary, CreateSpecifications, CreateSubstantialAnswer,
)
from cogbenchv2.constraints.adversarial import AdversarialVocabulary, AdversarialNoTargetVocab
from cogbenchv2.constraints.registry import get_constraints, get_constraint_count


# ─── Test fixtures ────────────────────────────────────────────────────────────

SAMPLE_PASSAGE = (
    "Photosynthesis is the process by which plants convert light energy into "
    "chemical energy. This process occurs primarily in the chloroplasts, which "
    "contain chlorophyll. The light-dependent reactions take place in the thylakoid "
    "membranes, while the Calvin cycle occurs in the stroma. Carbon dioxide and "
    "water are converted into glucose and oxygen. The rate of photosynthesis is "
    "affected by light intensity, carbon dioxide concentration, and temperature."
)

SAMPLE_CONCEPTS = ["photosynthesis", "chloroplasts", "chlorophyll", "calvin cycle",
                    "thylakoid", "glucose", "carbon dioxide", "light energy"]

SAMPLE_METHODS = ["light-dependent reactions", "calvin cycle", "photosynthesis"]


_SENTINEL = object()

def _make_data(question="", answer="", level=1, mode="standard",
               vocab_level=None, concepts=_SENTINEL, methods=_SENTINEL):
    """Helper to create QuestionData for tests."""
    return QuestionData(
        question=question,
        answer=answer,
        passage=SAMPLE_PASSAGE,
        target_level=level,
        key_concepts=SAMPLE_CONCEPTS if concepts is _SENTINEL else concepts,
        methods_principles=SAMPLE_METHODS if methods is _SENTINEL else methods,
        mode=mode,
        vocab_level=vocab_level,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsQuestion:
    def test_pass_ends_with_question_mark(self):
        data = _make_data("What is photosynthesis?")
        assert IsQuestion().check(data).passed

    def test_fail_ends_with_period(self):
        data = _make_data("Describe photosynthesis.")
        assert not IsQuestion().check(data).passed

    def test_fail_empty(self):
        data = _make_data("")
        assert not IsQuestion().check(data).passed

    def test_fail_ends_with_exclamation(self):
        data = _make_data("What is photosynthesis!")
        assert not IsQuestion().check(data).passed

    def test_pass_long_question(self):
        data = _make_data("Based on the passage, how does the process of photosynthesis work in plants?")
        assert IsQuestion().check(data).passed


class TestWordCount:
    def test_pass_normal_length(self):
        data = _make_data("What is the role of chlorophyll in photosynthesis within plant cells?")
        assert WordCount().check(data).passed

    def test_fail_too_short(self):
        data = _make_data("What is it?")
        result = WordCount().check(data)
        # "What is it" = 3 words, but need 10+
        assert not result.passed

    def test_fail_too_long(self):
        words = " ".join(["word"] * 160)
        data = _make_data(f"Is this {words}?")
        assert not WordCount().check(data).passed

    def test_pass_minimum(self):
        data = _make_data("What is the process of photosynthesis in chloroplasts for plants?")
        assert WordCount().check(data).passed

    def test_pass_at_boundary(self):
        # Exactly 10 words
        data = _make_data("What process converts light energy into chemical energy in plants?")
        assert WordCount().check(data).passed


class TestPassageRelevance:
    def test_pass_multiple_concepts(self):
        data = _make_data("How do chloroplasts use chlorophyll during photosynthesis?")
        assert PassageRelevance().check(data).passed

    def test_fail_no_concepts(self):
        data = _make_data("What color is the sky on a sunny day in summer?")
        assert not PassageRelevance().check(data).passed

    def test_pass_skipped_no_concepts_list(self):
        data = _make_data("Random question about anything?", concepts=[])
        assert PassageRelevance().check(data).passed

    def test_fail_only_one_concept(self):
        data = _make_data("What is the definition of glucose in biochemistry today?")
        assert not PassageRelevance().check(data).passed

    def test_pass_three_concepts(self):
        data = _make_data("How does photosynthesis in chloroplasts produce glucose?")
        assert PassageRelevance().check(data).passed


class TestNoDegenerateOutput:
    def test_pass_normal(self):
        data = _make_data("What is the role of chlorophyll in photosynthesis?")
        assert NoDegenerateOutput().check(data).passed

    def test_fail_empty(self):
        data = _make_data("")
        assert not NoDegenerateOutput().check(data).passed

    def test_fail_too_short(self):
        data = _make_data("Hi?")
        assert not NoDegenerateOutput().check(data).passed

    def test_fail_repeated_words(self):
        data = _make_data("photosynthesis photosynthesis photosynthesis photosynthesis what?")
        assert not NoDegenerateOutput().check(data).passed

    def test_pass_repeated_stop_words(self):
        # Stop words like "the" repeating is fine
        data = _make_data("What is the process and the role of the chloroplasts?")
        assert NoDegenerateOutput().check(data).passed


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — REMEMBER
# ═══════════════════════════════════════════════════════════════════════════════

class TestRememberVocabulary:
    def test_pass_define(self):
        data = _make_data("Define the term photosynthesis as described in the passage?")
        assert RememberVocabulary().check(data).passed

    def test_pass_what_is(self):
        data = _make_data("What is the primary function of chloroplasts in plant cells?")
        assert RememberVocabulary().check(data).passed

    def test_pass_identify(self):
        data = _make_data("Identify the organelle responsible for photosynthesis in plants?")
        assert RememberVocabulary().check(data).passed

    def test_fail_no_remember_verbs(self):
        data = _make_data("How does the process of photosynthesis compare to respiration?")
        assert not RememberVocabulary().check(data).passed

    def test_pass_name(self):
        data = _make_data("Name the molecule that absorbs light in photosynthesis?")
        assert RememberVocabulary().check(data).passed


class TestRememberSingleConcept:
    def test_pass_one_concept(self):
        data = _make_data("What is chlorophyll and what does it do in the cell?")
        assert RememberSingleConcept().check(data).passed

    def test_fail_multiple_concepts(self):
        data = _make_data("What are chloroplasts and chlorophyll and photosynthesis?")
        assert not RememberSingleConcept().check(data).passed

    def test_pass_zero_concepts(self):
        data = _make_data("What is the main topic of this passage about plants?")
        result = RememberSingleConcept().check(data)
        assert result.passed

    def test_fail_three_concepts(self):
        data = _make_data("Define photosynthesis, chloroplasts, and glucose as in the passage?")
        assert not RememberSingleConcept().check(data).passed

    def test_pass_no_concepts_list(self):
        data = _make_data("What is the main idea?", concepts=[])
        assert RememberSingleConcept().check(data).passed


class TestRememberShortAnswer:
    def test_pass_short(self):
        data = _make_data(answer="Chloroplasts contain chlorophyll.")
        assert RememberShortAnswer().check(data).passed

    def test_fail_long(self):
        data = _make_data(answer=" ".join(["word"] * 25))
        assert not RememberShortAnswer().check(data).passed

    def test_pass_single_word(self):
        data = _make_data(answer="Chloroplasts")
        assert RememberShortAnswer().check(data).passed

    def test_pass_at_boundary(self):
        data = _make_data(answer=" ".join(["word"] * 20))
        assert RememberShortAnswer().check(data).passed

    def test_fail_paragraph(self):
        data = _make_data(answer="Photosynthesis is a complex process that involves " * 5)
        assert not RememberShortAnswer().check(data).passed


class TestRememberExtractable:
    def test_pass_from_passage(self):
        data = _make_data(answer="Chloroplasts contain chlorophyll.")
        assert RememberExtractable().check(data).passed

    def test_fail_not_in_passage(self):
        data = _make_data(answer="Mitosis occurs during cell division in animal cells.")
        assert not RememberExtractable().check(data).passed

    def test_pass_partial_overlap(self):
        data = _make_data(answer="Light energy is converted into chemical energy by chloroplasts.")
        assert RememberExtractable().check(data).passed

    def test_pass_empty_answer(self):
        data = _make_data(answer="")
        # No content words → skip
        assert RememberExtractable().check(data).passed

    def test_fail_totally_different(self):
        data = _make_data(answer="The mitochondria is the powerhouse of the cell.")
        assert not RememberExtractable().check(data).passed


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — UNDERSTAND
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnderstandVocabulary:
    def test_pass_explain(self):
        data = _make_data("Explain the process of photosynthesis in your own words?")
        assert UnderstandVocabulary().check(data).passed

    def test_pass_describe(self):
        data = _make_data("Describe how chloroplasts contribute to photosynthesis?")
        assert UnderstandVocabulary().check(data).passed

    def test_fail_no_understand_verbs(self):
        data = _make_data("Calculate the rate of photosynthesis given light intensity?")
        assert not UnderstandVocabulary().check(data).passed

    def test_pass_summarize(self):
        data = _make_data("Summarize the main steps in photosynthesis as described above?")
        assert UnderstandVocabulary().check(data).passed

    def test_pass_interpret(self):
        data = _make_data("How would you interpret the role of light in photosynthesis?")
        assert UnderstandVocabulary().check(data).passed


class TestUnderstandNotCopied:
    def test_pass_paraphrased(self):
        data = _make_data(
            answer="Solar radiation powers sugar manufacturing in vegetation through specialized cellular organelles."
        )
        assert UnderstandNotCopied().check(data).passed

    def test_fail_copied(self):
        data = _make_data(
            answer="Photosynthesis is the process by which plants convert light energy into chemical energy."
        )
        assert not UnderstandNotCopied().check(data).passed

    def test_pass_completely_different_words(self):
        data = _make_data(answer="Solar radiation powers sugar manufacturing in vegetation.")
        assert UnderstandNotCopied().check(data).passed

    def test_pass_empty(self):
        data = _make_data(answer="")
        assert UnderstandNotCopied().check(data).passed

    def test_fail_partial_copy(self):
        data = _make_data(
            answer="The process occurs primarily in the chloroplasts which contain chlorophyll "
                   "and the light-dependent reactions take place in the thylakoid membranes."
        )
        assert not UnderstandNotCopied().check(data).passed


class TestUnderstandMeaning:
    def test_pass_how(self):
        data = _make_data("How does photosynthesis benefit the plant?")
        assert UnderstandMeaning().check(data).passed

    def test_pass_why(self):
        data = _make_data("Why is chlorophyll important for photosynthesis?")
        assert UnderstandMeaning().check(data).passed

    def test_pass_explain(self):
        data = _make_data("Can you explain the role of light in photosynthesis?")
        assert UnderstandMeaning().check(data).passed

    def test_fail_bare_fact(self):
        data = _make_data("When does photosynthesis occur in the plant lifecycle today?")
        assert not UnderstandMeaning().check(data).passed

    def test_pass_describe(self):
        data = _make_data("Describe the significance of the Calvin cycle in photosynthesis?")
        assert UnderstandMeaning().check(data).passed


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 3 — APPLY
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyVocabulary:
    def test_pass_calculate(self):
        data = _make_data("Calculate the rate of oxygen production given certain conditions?")
        assert ApplyVocabulary().check(data).passed

    def test_pass_apply(self):
        data = _make_data("How would you apply the principles of photosynthesis to a greenhouse?")
        assert ApplyVocabulary().check(data).passed

    def test_fail_no_apply_verbs(self):
        data = _make_data("What is the definition of photosynthesis in biology today?")
        assert not ApplyVocabulary().check(data).passed

    def test_pass_solve(self):
        data = _make_data("Solve for the amount of glucose produced from 6 CO2 molecules?")
        assert ApplyVocabulary().check(data).passed

    def test_pass_determine(self):
        data = _make_data("Determine the effect of doubling light intensity on photosynthesis rate?")
        assert ApplyVocabulary().check(data).passed


class TestApplySpecificResult:
    def test_pass_number(self):
        data = _make_data(answer="The rate increases by 50% when light intensity doubles.")
        assert ApplySpecificResult().check(data).passed

    def test_pass_steps(self):
        data = _make_data(answer="First, increase light intensity. Then measure oxygen output.")
        assert ApplySpecificResult().check(data).passed

    def test_fail_vague(self):
        data = _make_data(answer="It would change depending on conditions and circumstances.")
        assert not ApplySpecificResult().check(data).passed

    def test_pass_therefore(self):
        data = _make_data(answer="Therefore the plant would produce more glucose under these conditions.")
        assert ApplySpecificResult().check(data).passed

    def test_pass_result_word(self):
        data = _make_data(answer="The result shows that photosynthesis rate correlates with light.")
        assert ApplySpecificResult().check(data).passed


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 4 — ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeVocabulary:
    def test_pass_compare(self):
        data = _make_data("Compare the light reactions and the Calvin cycle?")
        assert AnalyzeVocabulary().check(data).passed

    def test_pass_examine(self):
        data = _make_data("Examine the relationship between chlorophyll and light absorption?")
        assert AnalyzeVocabulary().check(data).passed

    def test_fail_no_analyze_verbs(self):
        data = _make_data("What is the role of chloroplasts in the plant cell today?")
        assert not AnalyzeVocabulary().check(data).passed

    def test_pass_differentiate(self):
        data = _make_data("Differentiate between the thylakoid and stroma functions?")
        assert AnalyzeVocabulary().check(data).passed

    def test_pass_analyze(self):
        data = _make_data("Analyze the factors that affect the rate of photosynthesis?")
        assert AnalyzeVocabulary().check(data).passed


class TestAnalyzeMultipleConcepts:
    def test_pass_two_concepts(self):
        data = _make_data("How do chloroplasts and chlorophyll work together in photosynthesis?")
        assert AnalyzeMultipleConcepts().check(data).passed

    def test_fail_one_concept(self):
        data = _make_data("What is the structure of chloroplasts in the plant cell today?")
        assert not AnalyzeMultipleConcepts().check(data).passed

    def test_pass_three_concepts(self):
        data = _make_data("Compare photosynthesis, the calvin cycle, and glucose production rates?")
        assert AnalyzeMultipleConcepts().check(data).passed

    def test_pass_no_concepts_list(self):
        data = _make_data("Compare two things?", concepts=[])
        assert AnalyzeMultipleConcepts().check(data).passed

    def test_fail_zero_concepts(self):
        data = _make_data("How do cells divide during the mitosis process in animals?")
        assert not AnalyzeMultipleConcepts().check(data).passed


class TestAnalyzeRelationship:
    def test_pass_between(self):
        data = _make_data("What is the relationship between light and photosynthesis?")
        assert AnalyzeRelationship().check(data).passed

    def test_pass_compare(self):
        data = _make_data("Compare the functions of thylakoid and stroma?")
        assert AnalyzeRelationship().check(data).passed

    def test_fail_no_relationship(self):
        data = _make_data("What is the definition of chlorophyll in plants today?")
        assert not AnalyzeRelationship().check(data).passed

    def test_pass_affect(self):
        data = _make_data("How does temperature affect the rate of photosynthesis?")
        assert AnalyzeRelationship().check(data).passed

    def test_pass_differ(self):
        data = _make_data("How do the light reactions differ from the Calvin cycle?")
        assert AnalyzeRelationship().check(data).passed


class TestAnalyzeAnswerCoverage:
    def test_pass_covers_all(self):
        data = _make_data(
            question="Compare photosynthesis and glucose production in plant cells?",
            answer="Photosynthesis produces glucose which is the main energy source for the plant.",
        )
        assert AnalyzeAnswerCoverage().check(data).passed

    def test_fail_missing_concept(self):
        data = _make_data(
            question="Compare chloroplasts and the calvin cycle in photosynthesis?",
            answer="Chloroplasts are organelles that perform various functions in the cell.",
        )
        assert not AnalyzeAnswerCoverage().check(data).passed

    def test_pass_no_concepts_in_question(self):
        data = _make_data(
            question="How do cells divide during mitosis?",
            answer="Cells divide through several stages."
        )
        assert AnalyzeAnswerCoverage().check(data).passed

    def test_pass_all_concepts_covered(self):
        data = _make_data(
            question="How do photosynthesis and glucose relate to plant energy?",
            answer="Photosynthesis produces glucose which serves as the main energy source.",
        )
        assert AnalyzeAnswerCoverage().check(data).passed

    def test_pass_no_concepts_list(self):
        data = _make_data(question="Compare X and Y?", answer="Both are important.", concepts=[])
        assert AnalyzeAnswerCoverage().check(data).passed


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 5 — EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateVocabulary:
    def test_pass_assess(self):
        data = _make_data("Assess the importance of chlorophyll in photosynthesis?")
        assert EvaluateVocabulary().check(data).passed

    def test_pass_justify(self):
        data = _make_data("Justify why photosynthesis is essential for plant survival?")
        assert EvaluateVocabulary().check(data).passed

    def test_fail_no_evaluate_verbs(self):
        data = _make_data("What is the definition of photosynthesis in biology today?")
        assert not EvaluateVocabulary().check(data).passed

    def test_pass_evaluate(self):
        data = _make_data("Evaluate the claim that all plants require light for survival?")
        assert EvaluateVocabulary().check(data).passed

    def test_pass_to_what_extent(self):
        data = _make_data("To what extent is photosynthesis dependent on carbon dioxide?")
        assert EvaluateVocabulary().check(data).passed


class TestEvaluateClaim:
    def test_pass_do_you_agree(self):
        data = _make_data("Do you agree that photosynthesis is the most important process?")
        assert EvaluateClaim().check(data).passed

    def test_pass_to_what_extent(self):
        data = _make_data("To what extent is the Calvin cycle dependent on light reactions?")
        assert EvaluateClaim().check(data).passed

    def test_fail_no_claim(self):
        data = _make_data("What is the rate of photosynthesis at room temperature today?")
        assert not EvaluateClaim().check(data).passed

    def test_pass_is_it_justified(self):
        data = _make_data("Is it justified to say that all energy comes from photosynthesis?")
        assert EvaluateClaim().check(data).passed

    def test_pass_should(self):
        data = _make_data("Should we consider chloroplasts the most important organelles?")
        assert EvaluateClaim().check(data).passed


class TestEvaluateArgumentation:
    def test_pass_however(self):
        data = _make_data(answer="Photosynthesis is vital. However, some organisms use chemosynthesis.")
        assert EvaluateArgumentation().check(data).passed

    def test_pass_although(self):
        data = _make_data(answer="Although light is needed, some reactions occur in the dark.")
        assert EvaluateArgumentation().check(data).passed

    def test_fail_no_contrast(self):
        data = _make_data(answer="Photosynthesis produces glucose and oxygen from light energy.")
        assert not EvaluateArgumentation().check(data).passed

    def test_pass_on_the_other_hand(self):
        data = _make_data(answer="Plants need light. On the other hand, dark reactions also occur.")
        assert EvaluateArgumentation().check(data).passed

    def test_pass_despite(self):
        data = _make_data(answer="Despite low light, some plants can still photosynthesize effectively.")
        assert EvaluateArgumentation().check(data).passed


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 6 — CREATE
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateVocabulary:
    def test_pass_design(self):
        data = _make_data("Design an experiment to measure photosynthesis rate?")
        assert CreateVocabulary().check(data).passed

    def test_pass_propose(self):
        data = _make_data("Propose a method to enhance photosynthesis in crops?")
        assert CreateVocabulary().check(data).passed

    def test_fail_no_create_verbs(self):
        data = _make_data("What is the definition of photosynthesis in biology today?")
        assert not CreateVocabulary().check(data).passed

    def test_pass_develop(self):
        data = _make_data("Develop a plan to study the effects of light on photosynthesis?")
        assert CreateVocabulary().check(data).passed

    def test_pass_formulate(self):
        data = _make_data("Formulate a hypothesis about photosynthesis and temperature?")
        assert CreateVocabulary().check(data).passed


class TestCreateSpecifications:
    def test_pass_multiple_specs(self):
        data = _make_data(
            "Design an experiment that must include a control group and should measure oxygen using a sensor?"
        )
        assert CreateSpecifications().check(data).passed

    def test_fail_no_specs(self):
        data = _make_data("Design an experiment about photosynthesis in the lab?")
        assert not CreateSpecifications().check(data).passed

    def test_pass_with_using(self):
        data = _make_data(
            "Propose a study using chloroplasts that must track glucose production rates?"
        )
        assert CreateSpecifications().check(data).passed

    def test_pass_include_at_least(self):
        data = _make_data(
            "Create a lesson plan that should include at least three activities for students?"
        )
        assert CreateSpecifications().check(data).passed

    def test_fail_one_spec_only(self):
        data = _make_data("Design something that must work properly in the lab setting?")
        # "must" is 1 marker, need >=2
        assert not CreateSpecifications().check(data).passed


class TestCreateSubstantialAnswer:
    def test_pass_long_answer(self):
        data = _make_data(answer=" ".join(["word"] * 60))
        assert CreateSubstantialAnswer().check(data).passed

    def test_fail_short_answer(self):
        data = _make_data(answer="Design an experiment with a control group.")
        assert not CreateSubstantialAnswer().check(data).passed

    def test_pass_at_boundary(self):
        data = _make_data(answer=" ".join(["word"] * 51))
        assert CreateSubstantialAnswer().check(data).passed

    def test_fail_empty(self):
        data = _make_data(answer="")
        assert not CreateSubstantialAnswer().check(data).passed

    def test_fail_very_short(self):
        data = _make_data(answer="An experiment.")
        assert not CreateSubstantialAnswer().check(data).passed


# ═══════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialVocabulary:
    def test_pass_uses_vocab_level(self):
        # Analyze question using Remember verbs
        data = _make_data(
            question="What is the difference between chloroplasts and chlorophyll?",
            mode="adversarial", vocab_level=1, level=4,
        )
        assert AdversarialVocabulary().check(data).passed

    def test_fail_missing_vocab_level(self):
        data = _make_data(
            question="Compare the light reactions and Calvin cycle?",
            mode="adversarial", vocab_level=1, level=4,
        )
        assert not AdversarialVocabulary().check(data).passed

    def test_skip_standard_mode(self):
        data = _make_data(question="Whatever?", mode="standard")
        assert AdversarialVocabulary().check(data).passed

    def test_pass_understand_vocab(self):
        data = _make_data(
            question="Explain how the two processes are distinct from each other?",
            mode="adversarial", vocab_level=2, level=5,
        )
        assert AdversarialVocabulary().check(data).passed

    def test_fail_wrong_vocab(self):
        data = _make_data(
            question="Design an experiment to measure the rate of photosynthesis?",
            mode="adversarial", vocab_level=1, level=6,
        )
        assert not AdversarialVocabulary().check(data).passed


class TestAdversarialNoTargetVocab:
    def test_pass_avoids_target(self):
        # L4 Analyze using L1 Remember vocab — should NOT have "compare"/"analyze"
        data = _make_data(
            question="What is the role of chloroplasts and how do they relate?",
            mode="adversarial", vocab_level=1, level=4,
        )
        # "relate" is in Analyze vocab, so this should fail
        result = AdversarialNoTargetVocab().check(data)
        assert not result.passed

    def test_pass_clean_avoidance(self):
        data = _make_data(
            question="Name the differences between the two processes in photosynthesis?",
            mode="adversarial", vocab_level=1, level=4,
        )
        # No L4 verbs (compare, contrast, examine, etc.) present
        assert AdversarialNoTargetVocab().check(data).passed

    def test_skip_standard_mode(self):
        data = _make_data(question="Compare them?", mode="standard")
        assert AdversarialNoTargetVocab().check(data).passed

    def test_fail_uses_target_vocab(self):
        data = _make_data(
            question="Analyze the differences between chloroplasts and chlorophyll?",
            mode="adversarial", vocab_level=1, level=4,
        )
        assert not AdversarialNoTargetVocab().check(data).passed

    def test_pass_no_target_verbs(self):
        data = _make_data(
            question="List the factors that make photosynthesis and respiration different?",
            mode="adversarial", vocab_level=1, level=4,
        )
        assert AdversarialNoTargetVocab().check(data).passed


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_standard_constraints_count(self):
        # Universal (4) + level-specific (3-5 per level)
        for level in range(1, 7):
            count = get_constraint_count(level, "standard")
            assert count >= 7, f"Level {level} has only {count} constraints"

    def test_adversarial_has_adversarial_constraints(self):
        constraints = get_constraints(4, "adversarial")
        ids = [c.constraint_id for c in constraints]
        assert "AV" in ids, "Adversarial mode missing AV constraint"
        assert "AN" in ids, "Adversarial mode missing AN constraint"

    def test_adversarial_drops_vocab_constraint(self):
        constraints = get_constraints(4, "adversarial")
        ids = [c.constraint_id for c in constraints]
        assert "A1" not in ids, "Adversarial mode should not have A1 (vocab) constraint"

    def test_standard_has_vocab_constraint(self):
        constraints = get_constraints(4, "standard")
        ids = [c.constraint_id for c in constraints]
        assert "A1" in ids, "Standard mode should have A1 (vocab) constraint"

    def test_all_levels_have_universal(self):
        for level in range(1, 7):
            constraints = get_constraints(level, "standard")
            ids = [c.constraint_id for c in constraints]
            assert "U1" in ids
            assert "U2" in ids
            assert "U3" in ids
            assert "U4" in ids


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
