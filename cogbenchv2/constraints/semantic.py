"""Tier 2 Semantic Constraints — NLI-based verification.

These constraints use a pre-trained NLI model (DeBERTa-v3-base on MNLI) to check:
- D3: Is the answer supported by (entailed by) the passage?
- P2: Is the scenario in the question NEW (not entailed by passage)?
- C2: Is the expected output NOT already in the passage?

The NLI model is NOT a judge — it's a verification tool with a fixed threshold.
Same input always produces the same output.
"""

import torch
from typing import Optional

from cogbenchv2.constraints.base import Constraint, QuestionData, ConstraintResult
from cogbenchv2.config import (
    NLI_MODEL_NAME, NLI_ENTAILMENT_THRESHOLD, NLI_NOVELTY_THRESHOLD, NLI_DEVICE,
)


# Lazy-loaded singleton for the NLI model
_nli_model = None
_nli_tokenizer = None


def get_nli_model():
    """Load NLI model (singleton — only loads once)."""
    global _nli_model, _nli_tokenizer

    if _nli_model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"Loading NLI model: {NLI_MODEL_NAME}...")
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)

        device = f"cuda:{NLI_DEVICE}" if torch.cuda.is_available() else "cpu"
        _nli_model = _nli_model.to(device)
        _nli_model.eval()
        print(f"  NLI model loaded on {device}")

    return _nli_model, _nli_tokenizer


def nli_predict(premise: str, hypothesis: str, max_length: int = 512):
    """Run NLI prediction. Returns dict with entailment/neutral/contradiction probs.

    Args:
        premise: The text that is assumed to be true (e.g., passage)
        hypothesis: The text to check against the premise (e.g., answer)

    Returns:
        {"entailment": float, "neutral": float, "contradiction": float}
    """
    model, tokenizer = get_nli_model()
    device = next(model.parameters()).device

    inputs = tokenizer(premise, hypothesis, return_tensors="pt",
                       truncation=True, max_length=max_length, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    # Map labels dynamically from model config (works for any NLI model)
    id2label = model.config.id2label
    result = {}
    for idx, label in id2label.items():
        label_clean = label.lower().strip()
        idx_int = int(idx)
        if "entail" in label_clean:
            result["entailment"] = probs[idx_int].item()
        elif "neutral" in label_clean:
            result["neutral"] = probs[idx_int].item()
        elif "contradict" in label_clean:
            result["contradiction"] = probs[idx_int].item()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# D3 — Understand: Answer is consistent with passage (not contradicted)
# ═══════════════════════════════════════════════════════════════════════════════

class UnderstandSupported(Constraint):
    """D3: Answer is consistent with the passage — NLI contradiction < 50%."""

    constraint_id = "D3"
    constraint_name = "answer_supported"
    tier = "semantic"

    CONTRADICTION_THRESHOLD = 0.50

    def check(self, data: QuestionData) -> ConstraintResult:
        if not data.passage or not data.answer:
            return self._result(True, 1.0, "Missing passage or answer; skipped")

        # Truncate passage to first ~400 words for NLI input
        passage_truncated = " ".join(data.passage.split()[:400])

        try:
            scores = nli_predict(passage_truncated, data.answer)
        except Exception as e:
            return self._result(False, 0.0, f"NLI error: {e}")

        contradiction = scores.get("contradiction", 1.0)
        passed = contradiction < self.CONTRADICTION_THRESHOLD

        if passed:
            return self._result(True, 1.0 - contradiction,
                                f"Answer consistent with passage: contradiction={contradiction:.3f} "
                                f"(threshold <{self.CONTRADICTION_THRESHOLD})")
        return self._result(False, 1.0 - contradiction,
                            f"Answer contradicts passage: contradiction={contradiction:.3f} "
                            f"(need <{self.CONTRADICTION_THRESHOLD}). "
                            "Understand answers must be consistent with the text.")


# ═══════════════════════════════════════════════════════════════════════════════
# P2 — Apply: Scenario is NEW (not in passage)
# ═══════════════════════════════════════════════════════════════════════════════

class ApplyNewScenario(Constraint):
    """P2: Question describes a NEW scenario not in the passage — NLI non-entailment >55%."""

    constraint_id = "P2"
    constraint_name = "new_scenario"
    tier = "semantic"

    def check(self, data: QuestionData) -> ConstraintResult:
        if not data.passage or not data.question:
            return self._result(True, 1.0, "Missing passage or question; skipped")

        passage_truncated = " ".join(data.passage.split()[:400])

        try:
            scores = nli_predict(passage_truncated, data.question)
        except Exception as e:
            return self._result(False, 0.0, f"NLI error: {e}")

        non_entailment = scores.get("neutral", 0.0) + scores.get("contradiction", 0.0)
        passed = non_entailment >= NLI_NOVELTY_THRESHOLD

        if passed:
            return self._result(True, non_entailment,
                                f"Question presents novel scenario: "
                                f"non-entailment={non_entailment:.3f} "
                                f"(threshold {NLI_NOVELTY_THRESHOLD})")
        return self._result(False, non_entailment,
                            f"Scenario too similar to passage: "
                            f"non-entailment={non_entailment:.3f} "
                            f"(need >={NLI_NOVELTY_THRESHOLD}). "
                            "Apply questions should present a new scenario.")


# ═══════════════════════════════════════════════════════════════════════════════
# C2 — Create: Expected output is NOT in the passage
# ═══════════════════════════════════════════════════════════════════════════════

class CreateNovel(Constraint):
    """C2: Asks for something NOT in the passage — NLI non-entailment >60%."""

    constraint_id = "C2"
    constraint_name = "novel_creation"
    tier = "semantic"

    # Slightly higher threshold than Apply — creation should be clearly novel
    NOVELTY_THRESHOLD = 0.60

    def check(self, data: QuestionData) -> ConstraintResult:
        if not data.passage or not data.answer:
            return self._result(True, 1.0, "Missing passage or answer; skipped")

        passage_truncated = " ".join(data.passage.split()[:400])

        try:
            scores = nli_predict(passage_truncated, data.answer)
        except Exception as e:
            return self._result(False, 0.0, f"NLI error: {e}")

        non_entailment = scores.get("neutral", 0.0) + scores.get("contradiction", 0.0)
        passed = non_entailment >= self.NOVELTY_THRESHOLD

        if passed:
            return self._result(True, non_entailment,
                                f"Answer is novel: non-entailment={non_entailment:.3f} "
                                f"(threshold {self.NOVELTY_THRESHOLD})")
        return self._result(False, non_entailment,
                            f"Answer too similar to passage: "
                            f"non-entailment={non_entailment:.3f} "
                            f"(need >={self.NOVELTY_THRESHOLD}). "
                            "Create answers should produce something new.")
