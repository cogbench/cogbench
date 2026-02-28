"""Run all constraints on generated questions and produce scored results."""

import os
import json
from datetime import datetime
from typing import Optional

from cogbenchv2.constraints.base import QuestionData, ConstraintResult
from cogbenchv2.constraints.registry import get_constraints
from cogbenchv2.config import RESULTS_DIR
from cogbenchv2.passages.processor import load_all_passages


def evaluate_question(question: str, answer: str, passage: dict,
                      level: int, mode: str = "standard",
                      vocab_level: Optional[int] = None) -> list:
    """Evaluate a single generated question against all applicable constraints.

    Args:
        question: Generated question text
        answer: Reference answer text
        passage: Passage dict with text, key_concepts, methods_principles
        level: Target Bloom's level (1-6)
        mode: "standard" or "adversarial"
        vocab_level: For adversarial mode, which level's vocabulary was required

    Returns:
        List of ConstraintResult objects
    """
    data = QuestionData(
        question=question or "",
        answer=answer or "",
        passage=passage.get("text", ""),
        target_level=level,
        key_concepts=passage.get("key_concepts", []),
        methods_principles=passage.get("methods_principles", []),
        mode=mode,
        vocab_level=vocab_level,
        subject=passage.get("subject", ""),
        passage_id=passage.get("passage_id", ""),
    )

    constraints = get_constraints(level, mode)
    results = []
    for c in constraints:
        try:
            result = c.check(data)
            results.append(result)
        except Exception as e:
            results.append(ConstraintResult(
                constraint_id=c.constraint_id,
                constraint_name=c.constraint_name,
                passed=False,
                score=0.0,
                details=f"Error: {str(e)}",
                tier=c.tier,
            ))

    return results


def evaluate_generations(generations_file: str, passages: list = None,
                         output_dir: str = None) -> list:
    """Evaluate all generations in a file.

    Args:
        generations_file: Path to generation results JSON
        passages: Pre-loaded passages (or loads from disk)
        output_dir: Where to save evaluation results

    Returns:
        List of evaluated records
    """
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Load generations
    with open(generations_file) as f:
        data = json.load(f)
    generations = data.get("generations", [])
    model = data.get("model", "unknown")
    mode = data.get("mode", "standard")

    print(f"\n  Evaluating {len(generations)} generations ({model}, {mode})")

    # Load passages
    if passages is None:
        passages = load_all_passages()
    passage_map = {p["passage_id"]: p for p in passages}

    evaluated = []
    n_pass = 0
    n_total = len(generations)

    for i, gen in enumerate(generations):
        passage_id = gen.get("passage_id", "")
        passage = passage_map.get(passage_id, {})

        if not gen.get("question"):
            # Generation failed — all constraints fail
            evaluated.append({
                **gen,
                "constraints": [],
                "all_passed": False,
                "pass_rate": 0.0,
                "evaluation_error": "No question generated",
            })
            continue

        results = evaluate_question(
            question=gen["question"],
            answer=gen.get("answer", ""),
            passage=passage,
            level=gen["level"],
            mode=gen.get("mode", "standard"),
            vocab_level=gen.get("vocab_level"),
        )

        all_passed = all(r.passed for r in results)
        pass_rate = sum(r.passed for r in results) / len(results) if results else 0.0
        if all_passed:
            n_pass += 1

        record = {
            **gen,
            "constraints": [
                {
                    "id": r.constraint_id,
                    "name": r.constraint_name,
                    "passed": r.passed,
                    "score": r.score,
                    "details": r.details,
                    "tier": r.tier,
                }
                for r in results
            ],
            "all_passed": all_passed,
            "pass_rate": round(pass_rate, 4),
        }
        evaluated.append(record)

        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            print(f"  [{i+1}/{n_total}] prompt-pass: {n_pass}/{i+1} "
                  f"({n_pass/(i+1)*100:.1f}%)")

    # Save
    save_name = os.path.basename(generations_file).replace("gen_", "eval_")
    save_path = os.path.join(output_dir, save_name)
    with open(save_path, "w") as f:
        json.dump({
            "model": model,
            "mode": mode,
            "n_evaluated": len(evaluated),
            "prompt_pass_rate": n_pass / n_total if n_total else 0,
            "timestamp": datetime.now().isoformat(),
            "evaluations": evaluated,
        }, f, indent=2)

    print(f"  Prompt-level strict: {n_pass}/{n_total} ({n_pass/n_total*100:.1f}%)")
    print(f"  Saved to: {save_path}")

    return evaluated
