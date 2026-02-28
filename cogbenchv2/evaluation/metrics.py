"""Compute benchmark metrics — IFEval-style strict/loose pass rates."""

import os
import json
import numpy as np
from collections import defaultdict
from typing import Optional

from cogbenchv2.config import (
    BLOOM_LEVELS, SUBJECTS, SEED, BOOTSTRAP_N, CONFIDENCE_LEVEL,
)


def compute_metrics(evaluations: list) -> dict:
    """Compute all benchmark metrics from evaluated records.

    Args:
        evaluations: List of evaluated records (from evaluate.py)

    Returns:
        Dict with all metrics
    """
    if not evaluations:
        return {}

    metrics = {
        "prompt_level_strict": _prompt_level(evaluations, loose=False),
        "prompt_level_loose": _prompt_level(evaluations, loose=True),
        "constraint_level_strict": _constraint_level(evaluations, loose=False),
        "constraint_level_loose": _constraint_level(evaluations, loose=True),
        "by_level": _by_bloom_level(evaluations),
        "by_subject": _by_subject(evaluations),
        "by_constraint": _by_constraint(evaluations),
        "by_tier": _by_tier(evaluations),
    }

    return metrics


def _prompt_level(evaluations: list, loose: bool = False) -> dict:
    """Prompt-level pass rate: % of questions where ALL constraints pass."""
    n_total = len(evaluations)
    if n_total == 0:
        return {"rate": 0.0, "n": 0, "ci_low": 0.0, "ci_high": 0.0}

    if loose:
        # Loose: use score >= 0.5 instead of strict pass
        passes = [all(c.get("score", 0) >= 0.5 for c in e.get("constraints", []))
                  for e in evaluations]
    else:
        passes = [e.get("all_passed", False) for e in evaluations]

    rate = sum(passes) / n_total
    ci_low, ci_high = _bootstrap_ci(passes)

    return {
        "rate": round(rate, 4),
        "n": n_total,
        "n_pass": sum(passes),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
    }


def _constraint_level(evaluations: list, loose: bool = False) -> dict:
    """Constraint-level pass rate: % of individual constraints that pass."""
    all_results = []
    for e in evaluations:
        for c in e.get("constraints", []):
            if loose:
                all_results.append(c.get("score", 0) >= 0.5)
            else:
                all_results.append(c.get("passed", False))

    n_total = len(all_results)
    if n_total == 0:
        return {"rate": 0.0, "n": 0, "ci_low": 0.0, "ci_high": 0.0}

    rate = sum(all_results) / n_total
    ci_low, ci_high = _bootstrap_ci(all_results)

    return {
        "rate": round(rate, 4),
        "n": n_total,
        "n_pass": sum(all_results),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
    }


def _by_bloom_level(evaluations: list) -> dict:
    """Break down metrics by Bloom's level."""
    by_level = defaultdict(list)
    for e in evaluations:
        by_level[e.get("level", 0)].append(e)

    result = {}
    for level in sorted(BLOOM_LEVELS.keys()):
        level_evals = by_level.get(level, [])
        if not level_evals:
            continue

        n_pass = sum(1 for e in level_evals if e.get("all_passed", False))
        n_total = len(level_evals)
        passes = [e.get("all_passed", False) for e in level_evals]

        result[level] = {
            "name": BLOOM_LEVELS[level],
            "prompt_strict": round(n_pass / n_total, 4) if n_total else 0,
            "n": n_total,
            "n_pass": n_pass,
            "ci_low": round(_bootstrap_ci(passes)[0], 4),
            "ci_high": round(_bootstrap_ci(passes)[1], 4),
        }

    return result


def _by_subject(evaluations: list) -> dict:
    """Break down metrics by subject."""
    by_subject = defaultdict(list)
    for e in evaluations:
        by_subject[e.get("subject", "unknown")].append(e)

    result = {}
    for subject in sorted(by_subject.keys()):
        evals = by_subject[subject]
        n_pass = sum(1 for e in evals if e.get("all_passed", False))
        n_total = len(evals)

        result[subject] = {
            "prompt_strict": round(n_pass / n_total, 4) if n_total else 0,
            "n": n_total,
            "n_pass": n_pass,
        }

    return result


def _by_constraint(evaluations: list) -> dict:
    """Break down pass rates per individual constraint."""
    constraint_results = defaultdict(lambda: {"pass": 0, "total": 0})

    for e in evaluations:
        for c in e.get("constraints", []):
            cid = c.get("id", "?")
            constraint_results[cid]["total"] += 1
            if c.get("passed", False):
                constraint_results[cid]["pass"] += 1

    result = {}
    for cid, counts in sorted(constraint_results.items()):
        rate = counts["pass"] / counts["total"] if counts["total"] else 0
        result[cid] = {
            "pass_rate": round(rate, 4),
            "n_pass": counts["pass"],
            "n_total": counts["total"],
        }

    return result


def _by_tier(evaluations: list) -> dict:
    """Break down pass rates by constraint tier."""
    tier_results = defaultdict(lambda: {"pass": 0, "total": 0})

    for e in evaluations:
        for c in e.get("constraints", []):
            tier = c.get("tier", "unknown")
            tier_results[tier]["total"] += 1
            if c.get("passed", False):
                tier_results[tier]["pass"] += 1

    result = {}
    for tier, counts in sorted(tier_results.items()):
        rate = counts["pass"] / counts["total"] if counts["total"] else 0
        result[tier] = {
            "pass_rate": round(rate, 4),
            "n_pass": counts["pass"],
            "n_total": counts["total"],
        }

    return result


def compute_adversarial_gap(standard_metrics: dict, adversarial_metrics: dict) -> dict:
    """Compute the adversarial gap — the headline metric.

    Gap = Standard pass rate - Adversarial pass rate
    """
    std_rate = standard_metrics.get("prompt_level_strict", {}).get("rate", 0)
    adv_rate = adversarial_metrics.get("prompt_level_strict", {}).get("rate", 0)
    gap = std_rate - adv_rate

    # Per-level gaps
    level_gaps = {}
    std_by_level = standard_metrics.get("by_level", {})
    adv_by_level = adversarial_metrics.get("by_level", {})

    for level in BLOOM_LEVELS:
        std_l = std_by_level.get(level, {}).get("prompt_strict", 0)
        adv_l = adv_by_level.get(level, {}).get("prompt_strict", 0)
        level_gaps[level] = {
            "name": BLOOM_LEVELS[level],
            "standard": round(std_l, 4),
            "adversarial": round(adv_l, 4),
            "gap": round(std_l - adv_l, 4),
        }

    return {
        "overall_gap": round(gap, 4),
        "standard_rate": round(std_rate, 4),
        "adversarial_rate": round(adv_rate, 4),
        "by_level": level_gaps,
    }


def _bootstrap_ci(values: list, n_bootstrap: int = BOOTSTRAP_N,
                  confidence: float = CONFIDENCE_LEVEL) -> tuple:
    """Compute bootstrap confidence interval for a list of 0/1 values."""
    if not values or len(values) < 2:
        mean = np.mean(values) if values else 0.0
        return (mean, mean)

    rng = np.random.RandomState(SEED)
    arr = np.array(values, dtype=float)
    boot_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    alpha = (1 - confidence) / 2
    ci_low = np.percentile(boot_means, alpha * 100)
    ci_high = np.percentile(boot_means, (1 - alpha) * 100)

    return (ci_low, ci_high)
