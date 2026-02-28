"""Run full benchmark for all local Ollama models sequentially.

Each model runs standard + adversarial modes on all 120 passages.
Results are saved incrementally so the run can be resumed.
"""

import sys
import os
import json
import time
import glob

from cogbenchv2.config import LOCAL_MODELS, BLOOM_LEVELS, ADVERSARIAL_PAIRINGS, RESULTS_DIR
from cogbenchv2.generation.generate import generate_for_model, unload_ollama_model
from cogbenchv2.evaluation.evaluate import evaluate_generations
from cogbenchv2.evaluation.metrics import compute_metrics, compute_adversarial_gap
from cogbenchv2.passages.processor import load_all_passages


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load passages
    passages = load_all_passages()
    if not passages:
        print("ERROR: No passages found. Run scrape_passages.py first.")
        sys.exit(1)

    print(f"Loaded {len(passages)} passages")
    models = list(LOCAL_MODELS.keys())
    modes = ["standard", "adversarial"]
    total_gens = len(models) * len(modes) * len(passages) * len(BLOOM_LEVELS)

    print(f"\n{'='*60}")
    print(f"  COGBENCH V2 — FULL LOCAL BENCHMARK")
    print(f"  Models: {len(models)} ({', '.join(models)})")
    print(f"  Passages: {len(passages)}")
    print(f"  Modes: {modes}")
    print(f"  Total generations: {total_gens}")
    print(f"{'='*60}\n")

    t_start = time.time()

    for i, model in enumerate(models):
        model_start = time.time()
        print(f"\n{'─'*60}")
        print(f"  MODEL {i+1}/{len(models)}: {model}")
        print(f"{'─'*60}")

        for mode in modes:
            generate_for_model(model, passages, mode=mode, output_dir=RESULTS_DIR)

        model_elapsed = time.time() - model_start
        print(f"\n  {model} complete in {model_elapsed/60:.1f} minutes")

    gen_elapsed = time.time() - t_start
    print(f"\n\n{'='*60}")
    print(f"  ALL GENERATION COMPLETE: {gen_elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    # Evaluate all
    print(f"\n{'='*60}")
    print(f"  EVALUATION")
    print(f"{'='*60}")

    gen_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "gen_*.json")))
    all_metrics = {}

    for gen_file in gen_files:
        basename = os.path.basename(gen_file)
        print(f"\n  Evaluating: {basename}")
        evaluated = evaluate_generations(gen_file, passages, RESULTS_DIR)
        metrics = compute_metrics(evaluated)
        key = basename.replace("gen_", "").replace(".json", "")
        all_metrics[key] = metrics

    # Adversarial gap per model
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")

    model_groups = {}
    for key, metrics in all_metrics.items():
        if key.endswith("_standard"):
            model_name = key.replace("_standard", "")
            model_groups.setdefault(model_name, {})["standard"] = metrics
        elif key.endswith("_adversarial"):
            model_name = key.replace("_adversarial", "")
            model_groups.setdefault(model_name, {})["adversarial"] = metrics

    print(f"\n  {'Model':<25s} {'Std Strict':>10s} {'Adv Strict':>10s} {'Gap':>8s}")
    print(f"  {'─'*55}")

    gaps = {}
    for model_name in sorted(model_groups.keys()):
        modes_dict = model_groups[model_name]
        if "standard" in modes_dict and "adversarial" in modes_dict:
            gap = compute_adversarial_gap(modes_dict["standard"], modes_dict["adversarial"])
            gaps[model_name] = gap
            std = gap["standard_rate"]
            adv = gap["adversarial_rate"]
            g = gap["overall_gap"]
            print(f"  {model_name:<25s} {std:>9.1%} {adv:>9.1%} {g:>7.1%}")

    # Save summary
    summary = {
        "metrics": {k: _serialize(v) for k, v in all_metrics.items()},
        "adversarial_gaps": gaps,
        "n_passages": len(passages),
        "models": list(LOCAL_MODELS.keys()),
        "total_time_minutes": round((time.time() - t_start) / 60, 1),
    }
    summary_path = os.path.join(RESULTS_DIR, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")
    print(f"  Total time: {(time.time() - t_start)/60:.1f} minutes")


def _serialize(obj):
    """Make metrics JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


if __name__ == "__main__":
    main()
