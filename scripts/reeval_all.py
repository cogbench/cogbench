"""Re-evaluate all completed models with fixed constraints.

Runs evaluation only (no generation) on existing gen files.
"""
import sys
import os
import json
import glob

from cogbenchv2.config import RESULTS_DIR
from cogbenchv2.evaluation.evaluate import evaluate_generations
from cogbenchv2.evaluation.metrics import compute_metrics, compute_adversarial_gap
from cogbenchv2.passages.processor import load_all_passages


def main():
    passages = load_all_passages()
    if not passages:
        print("ERROR: No passages found.")
        sys.exit(1)
    print(f"Loaded {len(passages)} passages")

    # Find all complete gen files (720 gens, 0 errors)
    gen_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "gen_*.json")))
    complete_files = []
    for gf in gen_files:
        data = json.load(open(gf))
        gens = data.get("generations", [])
        errors = sum(1 for g in gens if g.get("error"))
        ok = len(gens) - errors
        if ok >= 200:  # At least 200 successful generations
            complete_files.append(gf)
            print(f"  Will evaluate: {os.path.basename(gf)} ({ok}/{len(gens)} ok)")
        else:
            print(f"  Skipping: {os.path.basename(gf)} ({ok}/{len(gens)} ok)")

    # Evaluate each file
    print(f"\n{'='*60}")
    print(f"  RE-EVALUATION WITH FIXED CONSTRAINTS")
    print(f"{'='*60}")

    all_metrics = {}
    for gen_file in complete_files:
        print(f"\n  Evaluating: {os.path.basename(gen_file)}")
        evaluated = evaluate_generations(gen_file, passages, RESULTS_DIR)
        metrics = compute_metrics(evaluated)

        basename = os.path.basename(gen_file).replace("gen_", "").replace(".json", "")
        all_metrics[basename] = metrics

    # Adversarial gap
    print(f"\n{'='*60}")
    print(f"  ADVERSARIAL GAP ANALYSIS")
    print(f"{'='*60}")

    model_groups = {}
    for key, metrics in all_metrics.items():
        if key.endswith("_standard"):
            model_name = key.replace("_standard", "")
            model_groups.setdefault(model_name, {})["standard"] = metrics
        elif key.endswith("_adversarial"):
            model_name = key.replace("_adversarial", "")
            model_groups.setdefault(model_name, {})["adversarial"] = metrics

    for model_name, modes_dict in model_groups.items():
        if "standard" in modes_dict and "adversarial" in modes_dict:
            gap = compute_adversarial_gap(modes_dict["standard"], modes_dict["adversarial"])
            print(f"\n  {model_name}:")
            print(f"    Standard:    {gap['standard_rate']:.1%}")
            print(f"    Adversarial: {gap['adversarial_rate']:.1%}")
            print(f"    Gap:         {gap['overall_gap']:.1%}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"metrics": all_metrics}, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
