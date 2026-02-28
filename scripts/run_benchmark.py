"""Step 2: Run the full benchmark — generate + evaluate.

Usage:
    # Quick test: 1 model, standard mode
    python scripts/run_benchmark.py --model qwen2.5:14b --mode standard

    # Full run: all local models, both modes
    python scripts/run_benchmark.py --local-all --mode both

    # Evaluate only (skip generation)
    python scripts/run_benchmark.py --evaluate-only
"""

import sys
import os
import json
import glob
import argparse
from cogbenchv2.config import (
    LOCAL_MODELS, API_MODELS, TOGETHER_MODELS, RESULTS_DIR, BLOOM_LEVELS,
)
from cogbenchv2.generation.generate import generate_for_model
from cogbenchv2.evaluation.evaluate import evaluate_generations
from cogbenchv2.evaluation.metrics import compute_metrics, compute_adversarial_gap
from cogbenchv2.passages.processor import load_all_passages


def main():
    parser = argparse.ArgumentParser(description="CogBench Benchmark")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model to run")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["standard", "adversarial", "both"],
                        help="Generation mode")
    parser.add_argument("--local-all", action="store_true",
                        help="Run all local (Ollama) models")
    parser.add_argument("--api-all", action="store_true",
                        help="Run all API models")
    parser.add_argument("--together-all", action="store_true",
                        help="Run all Together.ai models")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip generation, only run evaluation")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    output_dir = args.output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Load passages
    passages = load_all_passages()
    if not passages:
        print("ERROR: No passages found. Run scrape_passages.py first.")
        sys.exit(1)
    print(f"Loaded {len(passages)} passages")

    # Determine models
    if args.local_all:
        models = list(LOCAL_MODELS.keys())
    elif args.api_all:
        models = list(API_MODELS.keys())
    elif args.together_all:
        models = list(TOGETHER_MODELS.keys())
    elif args.model:
        models = [args.model]
    else:
        models = list(LOCAL_MODELS.keys())

    modes = ["standard", "adversarial"] if args.mode == "both" else [args.mode]

    if not args.evaluate_only:
        # ─── Generation ──────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  GENERATION")
        print(f"  Models: {len(models)}, Modes: {modes}")
        print(f"  Passages: {len(passages)}, Levels: {len(BLOOM_LEVELS)}")
        total = len(models) * len(modes) * len(passages) * len(BLOOM_LEVELS)
        print(f"  Total generations: {total}")
        print(f"{'='*60}")

        for model in models:
            for mode in modes:
                generate_for_model(model, passages, mode=mode, output_dir=output_dir)

    # ─── Evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION")
    print(f"{'='*60}")

    gen_files = sorted(glob.glob(os.path.join(output_dir, "gen_*.json")))
    if not gen_files:
        print("No generation files found to evaluate.")
        return

    all_metrics = {}
    for gen_file in gen_files:
        print(f"\n  Evaluating: {os.path.basename(gen_file)}")
        evaluated = evaluate_generations(gen_file, passages, output_dir)
        metrics = compute_metrics(evaluated)

        # Extract model and mode from filename
        basename = os.path.basename(gen_file).replace("gen_", "").replace(".json", "")
        all_metrics[basename] = metrics

    # ─── Adversarial Gap ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ADVERSARIAL GAP ANALYSIS")
    print(f"{'='*60}")

    # Group metrics by model
    model_groups = {}
    for key, metrics in all_metrics.items():
        if key.endswith("_standard"):
            model_name = key.replace("_standard", "")
            model_groups.setdefault(model_name, {})["standard"] = metrics
        elif key.endswith("_adversarial"):
            model_name = key.replace("_adversarial", "")
            model_groups.setdefault(model_name, {})["adversarial"] = metrics

    gaps = {}
    for model_name, modes_dict in model_groups.items():
        if "standard" in modes_dict and "adversarial" in modes_dict:
            gap = compute_adversarial_gap(modes_dict["standard"], modes_dict["adversarial"])
            gaps[model_name] = gap
            print(f"\n  {model_name}:")
            print(f"    Standard:    {gap['standard_rate']:.1%}")
            print(f"    Adversarial: {gap['adversarial_rate']:.1%}")
            print(f"    Gap:         {gap['overall_gap']:.1%}")

    # Save all metrics
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "metrics": all_metrics,
            "adversarial_gaps": gaps,
        }, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
