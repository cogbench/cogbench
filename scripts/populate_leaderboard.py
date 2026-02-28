"""Evaluate completed models and populate leaderboard data.json.

Run this whenever new model results are available — it only processes
models that have both standard and adversarial generation files complete.
"""

import os
import sys
import json
import glob
from datetime import datetime

from cogbenchv2.evaluation.evaluate import evaluate_generations
from cogbenchv2.evaluation.metrics import compute_metrics, compute_adversarial_gap
from cogbenchv2.passages.processor import load_all_passages
from cogbenchv2.config import RESULTS_DIR, BLOOM_LEVELS

# Model display names
MODEL_NAMES = {
    "qwen2.5:14b": ("Qwen 2.5 14B", "14B", "local"),
    "llama3.1:8b": ("Llama 3.1 8B", "8B", "local"),
    "gemma3:12b": ("Gemma 3 12B", "12B", "local"),
    "phi4:14b": ("Phi-4 14B", "14B", "local"),
    "deepseek-r1:14b": ("DeepSeek-R1 14B", "14B", "local"),
    "mistral-nemo:12b": ("Mistral Nemo 12B", "12B", "local"),
    "gemma2:9b": ("Gemma 2 9B", "9B", "local"),
}

LEADERBOARD_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "leaderboard", "data.json"),
    "/home/aietlab/Mourya/Benchmark/cogbench/leaderboard/data.json",
]


def find_completed_models():
    """Find models that have both standard and adversarial gen files with low error rates."""
    gen_files = glob.glob(os.path.join(RESULTS_DIR, "gen_*.json"))
    models = {}

    for f in gen_files:
        basename = os.path.basename(f)
        # Parse: gen_{model_safe}_{mode}.json
        parts = basename.replace("gen_", "").replace(".json", "")
        if parts.endswith("_standard"):
            mode = "standard"
            model_safe = parts[:-9]
        elif parts.endswith("_adversarial"):
            mode = "adversarial"
            model_safe = parts[:-12]
        else:
            continue

        if model_safe not in models:
            models[model_safe] = {}
        models[model_safe][mode] = f

    # Filter to models with both modes and acceptable error rates
    ready = {}
    for model_safe, files in models.items():
        if "standard" not in files or "adversarial" not in files:
            continue

        # Check error rates
        ok = True
        for mode, path in files.items():
            with open(path) as fh:
                data = json.load(fh)
            gens = data.get("generations", [])
            errors = sum(1 for g in gens if g.get("error"))
            error_rate = errors / len(gens) if gens else 1.0
            if error_rate > 0.1:  # >10% errors = not ready
                print(f"  Skipping {model_safe} ({mode}): {errors}/{len(gens)} errors ({error_rate*100:.0f}%)")
                ok = False
                break

        if ok:
            ready[model_safe] = files

    return ready


def model_safe_to_ollama(model_safe):
    """Convert gen file model name back to Ollama model name."""
    # gen file uses: model.replace(":", "_").replace(".", "_").replace("/", "_")
    # Reverse common patterns
    mappings = {
        "qwen2_5_14b": "qwen2.5:14b",
        "llama3_1_8b": "llama3.1:8b",
        "gemma3_12b": "gemma3:12b",
        "phi4_14b": "phi4:14b",
        "deepseek-r1_14b": "deepseek-r1:14b",
        "mistral-nemo_12b": "mistral-nemo:12b",
        "gemma2_9b": "gemma2:9b",
    }
    return mappings.get(model_safe, model_safe)


def main():
    print("=" * 60)
    print("CogBench — Leaderboard Population")
    print("=" * 60)

    # Find ready models
    ready = find_completed_models()
    if not ready:
        print("\nNo models ready (need both standard + adversarial with <10% errors)")
        return

    print(f"\nReady models: {list(ready.keys())}")

    # Load passages once
    print("\nLoading passages...")
    passages = load_all_passages()
    print(f"  {len(passages)} passages loaded")

    leaderboard = []
    per_level_data = {}
    all_constraint_results = {}
    model_details = {}

    for model_safe, files in sorted(ready.items()):
        ollama_name = model_safe_to_ollama(model_safe)
        display_name, params, model_type = MODEL_NAMES.get(
            ollama_name, (ollama_name, "?", "local"))

        print(f"\n{'─' * 50}")
        print(f"Model: {display_name}")

        # Check if eval files already exist
        std_eval_path = os.path.join(RESULTS_DIR, f"eval_{model_safe}_standard.json")
        adv_eval_path = os.path.join(RESULTS_DIR, f"eval_{model_safe}_adversarial.json")

        # Evaluate standard
        if os.path.exists(std_eval_path):
            print(f"  Loading existing evaluation: {std_eval_path}")
            with open(std_eval_path) as f:
                std_evals = json.load(f).get("evaluations", [])
        else:
            std_evals = evaluate_generations(files["standard"], passages)

        # Evaluate adversarial
        if os.path.exists(adv_eval_path):
            print(f"  Loading existing evaluation: {adv_eval_path}")
            with open(adv_eval_path) as f:
                adv_evals = json.load(f).get("evaluations", [])
        else:
            adv_evals = evaluate_generations(files["adversarial"], passages)

        # Compute metrics
        std_metrics = compute_metrics(std_evals)
        adv_metrics = compute_metrics(adv_evals)
        gap_data = compute_adversarial_gap(std_metrics, adv_metrics)

        # Print summary
        std_rate = std_metrics["prompt_level_strict"]["rate"]
        adv_rate = adv_metrics["prompt_level_strict"]["rate"]
        constraint_rate = std_metrics["constraint_level_strict"]["rate"]

        print(f"  Standard CSR:    {std_rate*100:.1f}%")
        print(f"  Adversarial CSR: {adv_rate*100:.1f}%")
        print(f"  Gap:             {gap_data['overall_gap']*100:.1f}pp")
        print(f"  Constraint %:    {constraint_rate*100:.1f}%")

        # Add to leaderboard
        leaderboard.append({
            "model_name": display_name,
            "model_params": params,
            "model_type": model_type,
            "standard_csr": std_rate,
            "adversarial_csr": adv_rate,
            "gap": gap_data["overall_gap"],
            "constraint_strict": constraint_rate,
            "prompt_strict": std_rate,
            "prompt_loose": std_metrics["prompt_level_loose"]["rate"],
            "constraint_loose": std_metrics["constraint_level_loose"]["rate"],
        })

        # Per-level data (standard)
        per_level_data[display_name] = {}
        for level in BLOOM_LEVELS:
            level_data = std_metrics.get("by_level", {}).get(level, {})
            per_level_data[display_name][level] = level_data.get("prompt_strict", 0)

        # ── Per-model detail data for clickable modal ──
        # Per-level: standard vs adversarial
        detail_per_level = {"standard": {}, "adversarial": {}}
        for level in BLOOM_LEVELS:
            std_l = std_metrics.get("by_level", {}).get(level, {})
            adv_l = adv_metrics.get("by_level", {}).get(level, {})
            detail_per_level["standard"][str(level)] = std_l.get("prompt_strict", 0)
            detail_per_level["adversarial"][str(level)] = adv_l.get("prompt_strict", 0)

        # Per-constraint pass rates (this model only)
        detail_constraints = {}
        for cid, cdata in sorted(std_metrics.get("by_constraint", {}).items()):
            rate = cdata.get("pass_rate", 0)
            detail_constraints[cid] = round(rate, 4)

        # Per-subject pass rates
        detail_subjects = {}
        for subj, sdata in sorted(std_metrics.get("by_subject", {}).items()):
            detail_subjects[subj] = sdata.get("prompt_strict", 0)

        # Per-tier pass rates
        detail_tiers = {}
        for tier, tdata in sorted(std_metrics.get("by_tier", {}).items()):
            detail_tiers[tier] = round(tdata.get("pass_rate", 0), 4)

        # Adversarial per-constraint
        detail_adv_constraints = {}
        for cid, cdata in sorted(adv_metrics.get("by_constraint", {}).items()):
            rate = cdata.get("pass_rate", 0)
            detail_adv_constraints[cid] = round(rate, 4)

        model_details[display_name] = {
            "per_level": detail_per_level,
            "per_constraint": detail_constraints,
            "per_constraint_adv": detail_adv_constraints,
            "per_subject": detail_subjects,
            "per_tier": detail_tiers,
            "prompt_strict": std_rate,
            "prompt_loose": std_metrics["prompt_level_loose"]["rate"],
            "constraint_strict": constraint_rate,
            "constraint_loose": std_metrics["constraint_level_loose"]["rate"],
            "adversarial_csr": adv_rate,
            "gap": gap_data["overall_gap"],
            "n_standard": std_metrics["prompt_level_strict"].get("n", 0),
            "n_adversarial": adv_metrics["prompt_level_strict"].get("n", 0),
        }

        # Constraint breakdown (aggregate across all models)
        for cid, cdata in std_metrics.get("by_constraint", {}).items():
            if cid not in all_constraint_results:
                all_constraint_results[cid] = {"pass": 0, "total": 0}
            all_constraint_results[cid]["pass"] += cdata.get("n_pass", 0)
            all_constraint_results[cid]["total"] += cdata.get("n_total", 0)

    # Sort leaderboard by gap (lower is better)
    leaderboard.sort(key=lambda r: r["gap"])

    # Compute constraint breakdown rates
    constraint_breakdown = {}
    for cid in sorted(all_constraint_results.keys()):
        c = all_constraint_results[cid]
        constraint_breakdown[cid] = round(c["pass"] / c["total"], 4) if c["total"] else 0

    # Build data.json
    data_json = {
        "stats": {
            "passages": 120,
            "subjects": 8,
            "bloom_levels": 6,
            "constraints": 28,
            "models_evaluated": len(leaderboard),
            "questions_per_model": 1440,
            "adversarial_pairings": 6,
        },
        "leaderboard": leaderboard,
        "per_level": per_level_data,
        "constraint_breakdown": constraint_breakdown,
        "model_details": model_details,
        "metadata": {
            "version": "1.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "note": f"Auto-generated from {len(leaderboard)} evaluated models",
        },
    }

    # Save to both locations
    for path in LEADERBOARD_PATHS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data_json, f, indent=2)
        print(f"\nSaved: {path}")

    # Print final leaderboard
    print(f"\n{'=' * 60}")
    print("LEADERBOARD")
    print(f"{'=' * 60}")
    print(f"{'Rank':<5} {'Model':<25} {'Std CSR':>8} {'Adv CSR':>8} {'Gap':>8} {'Constr%':>8}")
    print("-" * 65)
    for i, r in enumerate(leaderboard):
        print(f"{i+1:<5} {r['model_name']:<25} "
              f"{r['standard_csr']*100:>7.1f}% "
              f"{r['adversarial_csr']*100:>7.1f}% "
              f"{r['gap']*100:>7.1f}pp "
              f"{r['constraint_strict']*100:>7.1f}%")


if __name__ == "__main__":
    main()
