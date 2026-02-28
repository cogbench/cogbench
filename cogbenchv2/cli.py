"""CogBench CLI — command-line interface for the benchmark.

Usage:
    cogbench run --model qwen2.5:14b --mode both
    cogbench evaluate
    cogbench leaderboard
    cogbench scrape --subjects biology chemistry
    cogbench info
"""

import argparse
import sys


def cmd_run(args):
    """Generate questions and/or evaluate models."""
    import os
    import glob
    import json
    from cogbenchv2.config import (
        LOCAL_MODELS, API_MODELS, TOGETHER_MODELS, RESULTS_DIR, BLOOM_LEVELS,
    )
    from cogbenchv2.generation.generate import generate_for_model
    from cogbenchv2.evaluation.evaluate import evaluate_generations
    from cogbenchv2.evaluation.metrics import compute_metrics, compute_adversarial_gap
    from cogbenchv2.passages.processor import load_all_passages

    output_dir = args.output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    passages = load_all_passages()
    if not passages:
        print("ERROR: No passages found. Run 'cogbench scrape' first.")
        sys.exit(1)
    print(f"Loaded {len(passages)} passages")

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

    # Evaluation
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
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"metrics": all_metrics}, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")


def cmd_evaluate(args):
    """Re-evaluate all completed models with current constraints."""
    import os
    import json
    import glob
    from cogbenchv2.config import RESULTS_DIR
    from cogbenchv2.evaluation.evaluate import evaluate_generations
    from cogbenchv2.evaluation.metrics import compute_metrics, compute_adversarial_gap
    from cogbenchv2.passages.processor import load_all_passages

    passages = load_all_passages()
    if not passages:
        print("ERROR: No passages found.")
        sys.exit(1)
    print(f"Loaded {len(passages)} passages")

    gen_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "gen_*.json")))
    complete_files = []
    for gf in gen_files:
        with open(gf) as f:
            data = json.load(f)
        gens = data.get("generations", [])
        errors = sum(1 for g in gens if g.get("error"))
        ok = len(gens) - errors
        if ok >= 200:
            complete_files.append(gf)
            print(f"  Will evaluate: {os.path.basename(gf)} ({ok}/{len(gens)} ok)")
        else:
            print(f"  Skipping: {os.path.basename(gf)} ({ok}/{len(gens)} ok)")

    print(f"\n{'='*60}")
    print(f"  RE-EVALUATION WITH CURRENT CONSTRAINTS")
    print(f"{'='*60}")

    all_metrics = {}
    for gen_file in complete_files:
        print(f"\n  Evaluating: {os.path.basename(gen_file)}")
        evaluated = evaluate_generations(gen_file, passages, RESULTS_DIR)
        metrics = compute_metrics(evaluated)
        basename = os.path.basename(gen_file).replace("gen_", "").replace(".json", "")
        all_metrics[basename] = metrics

    # Adversarial gap
    model_groups = {}
    for key, metrics in all_metrics.items():
        if key.endswith("_standard"):
            model_name = key.replace("_standard", "")
            model_groups.setdefault(model_name, {})[" standard"] = metrics
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

    summary_path = os.path.join(RESULTS_DIR, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"metrics": all_metrics}, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")


def cmd_leaderboard(args):
    """Populate the leaderboard data.json from evaluation results."""
    import os
    import json
    import glob
    from datetime import datetime
    from cogbenchv2.evaluation.evaluate import evaluate_generations
    from cogbenchv2.evaluation.metrics import compute_metrics, compute_adversarial_gap
    from cogbenchv2.passages.processor import load_all_passages
    from cogbenchv2.config import RESULTS_DIR, BLOOM_LEVELS

    MODEL_NAMES = {
        "qwen2.5:14b": ("Qwen 2.5 14B", "14B", "local"),
        "llama3.1:8b": ("Llama 3.1 8B", "8B", "local"),
        "gemma3:12b": ("Gemma 3 12B", "12B", "local"),
        "phi4:14b": ("Phi-4 14B", "14B", "local"),
        "deepseek-r1:14b": ("DeepSeek-R1 14B", "14B", "local"),
        "mistral-nemo:12b": ("Mistral Nemo 12B", "12B", "local"),
        "gemma2:9b": ("Gemma 2 9B", "9B", "local"),
    }

    MAPPINGS = {
        "qwen2_5_14b": "qwen2.5:14b",
        "llama3_1_8b": "llama3.1:8b",
        "gemma3_12b": "gemma3:12b",
        "phi4_14b": "phi4:14b",
        "deepseek-r1_14b": "deepseek-r1:14b",
        "mistral-nemo_12b": "mistral-nemo:12b",
        "gemma2_9b": "gemma2:9b",
    }

    # Find completed models
    gen_files = glob.glob(os.path.join(RESULTS_DIR, "gen_*.json"))
    models = {}
    for f in gen_files:
        basename = os.path.basename(f)
        parts = basename.replace("gen_", "").replace(".json", "")
        if parts.endswith("_standard"):
            mode, model_safe = "standard", parts[:-9]
        elif parts.endswith("_adversarial"):
            mode, model_safe = "adversarial", parts[:-12]
        else:
            continue
        models.setdefault(model_safe, {})[mode] = f

    ready = {}
    for model_safe, files in models.items():
        if "standard" not in files or "adversarial" not in files:
            continue
        ok = True
        for mode, path in files.items():
            with open(path) as fh:
                data = json.load(fh)
            gens = data.get("generations", [])
            errors = sum(1 for g in gens if g.get("error"))
            if gens and errors / len(gens) > 0.1:
                ok = False
                break
        if ok:
            ready[model_safe] = files

    if not ready:
        print("No models ready (need both standard + adversarial with <10% errors)")
        return

    print(f"Ready models: {list(ready.keys())}")
    passages = load_all_passages()

    leaderboard = []
    per_level_data = {}
    model_details = {}
    all_constraint_results = {}

    for model_safe, files in sorted(ready.items()):
        ollama_name = MAPPINGS.get(model_safe, model_safe)
        display_name, params, model_type = MODEL_NAMES.get(
            ollama_name, (ollama_name, "?", "local"))
        print(f"\n  Model: {display_name}")

        std_eval_path = os.path.join(RESULTS_DIR, f"eval_{model_safe}_standard.json")
        adv_eval_path = os.path.join(RESULTS_DIR, f"eval_{model_safe}_adversarial.json")

        if os.path.exists(std_eval_path):
            with open(std_eval_path) as fh:
                std_evals = json.load(fh).get("evaluations", [])
        else:
            std_evals = evaluate_generations(files["standard"], passages)

        if os.path.exists(adv_eval_path):
            with open(adv_eval_path) as fh:
                adv_evals = json.load(fh).get("evaluations", [])
        else:
            adv_evals = evaluate_generations(files["adversarial"], passages)

        std_metrics = compute_metrics(std_evals)
        adv_metrics = compute_metrics(adv_evals)
        gap_data = compute_adversarial_gap(std_metrics, adv_metrics)

        std_rate = std_metrics["prompt_level_strict"]["rate"]
        adv_rate = adv_metrics["prompt_level_strict"]["rate"]
        constraint_rate = std_metrics["constraint_level_strict"]["rate"]

        leaderboard.append({
            "model_name": display_name, "model_params": params,
            "model_type": model_type, "standard_csr": std_rate,
            "adversarial_csr": adv_rate, "gap": gap_data["overall_gap"],
            "constraint_strict": constraint_rate, "prompt_strict": std_rate,
            "prompt_loose": std_metrics["prompt_level_loose"]["rate"],
            "constraint_loose": std_metrics["constraint_level_loose"]["rate"],
        })

        per_level_data[display_name] = {}
        for level in BLOOM_LEVELS:
            level_data = std_metrics.get("by_level", {}).get(level, {})
            per_level_data[display_name][level] = level_data.get("prompt_strict", 0)

        detail_per_level = {"standard": {}, "adversarial": {}}
        for level in BLOOM_LEVELS:
            std_l = std_metrics.get("by_level", {}).get(level, {})
            adv_l = adv_metrics.get("by_level", {}).get(level, {})
            detail_per_level["standard"][str(level)] = std_l.get("prompt_strict", 0)
            detail_per_level["adversarial"][str(level)] = adv_l.get("prompt_strict", 0)

        detail_constraints = {}
        for cid, cdata in sorted(std_metrics.get("by_constraint", {}).items()):
            detail_constraints[cid] = round(cdata.get("pass_rate", 0), 4)

        detail_subjects = {}
        for subj, sdata in sorted(std_metrics.get("by_subject", {}).items()):
            detail_subjects[subj] = sdata.get("prompt_strict", 0)

        detail_tiers = {}
        for tier, tdata in sorted(std_metrics.get("by_tier", {}).items()):
            detail_tiers[tier] = round(tdata.get("pass_rate", 0), 4)

        detail_adv_constraints = {}
        for cid, cdata in sorted(adv_metrics.get("by_constraint", {}).items()):
            detail_adv_constraints[cid] = round(cdata.get("pass_rate", 0), 4)

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

        for cid, cdata in std_metrics.get("by_constraint", {}).items():
            if cid not in all_constraint_results:
                all_constraint_results[cid] = {"pass": 0, "total": 0}
            all_constraint_results[cid]["pass"] += cdata.get("n_pass", 0)
            all_constraint_results[cid]["total"] += cdata.get("n_total", 0)

    leaderboard.sort(key=lambda r: r["gap"])

    constraint_breakdown = {}
    for cid in sorted(all_constraint_results.keys()):
        c = all_constraint_results[cid]
        constraint_breakdown[cid] = round(c["pass"] / c["total"], 4) if c["total"] else 0

    data_json = {
        "stats": {
            "passages": 120, "subjects": 8, "bloom_levels": 6,
            "constraints": 28, "models_evaluated": len(leaderboard),
            "questions_per_model": 1440, "adversarial_pairings": 6,
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

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data_json, f, indent=2)
    print(f"\n  Saved: {output_path}")

    print(f"\n{'='*60}")
    print(f"{'Rank':<5} {'Model':<25} {'Std CSR':>8} {'Adv CSR':>8} {'Gap':>8}")
    print("-" * 60)
    for i, r in enumerate(leaderboard):
        print(f"{i+1:<5} {r['model_name']:<25} "
              f"{r['standard_csr']*100:>7.1f}% "
              f"{r['adversarial_csr']*100:>7.1f}% "
              f"{r['gap']*100:>7.1f}pp")


def cmd_scrape(args):
    """Scrape textbook passages from OpenStax."""
    from cogbenchv2.passages.scraper import scrape_subject
    from cogbenchv2.passages.processor import process_all_passages
    from cogbenchv2.config import SUBJECTS, PASSAGES_PER_SUBJECT

    subjects = args.subjects or SUBJECTS
    n_passages = args.n_passages or PASSAGES_PER_SUBJECT

    print(f"Scraping {len(subjects)} subjects, {n_passages} passages each")
    for subject in subjects:
        scrape_subject(subject, n_passages)

    if not args.skip_process:
        print("\nProcessing passages (extracting key concepts)...")
        process_all_passages()
    print("\nDone!")


def cmd_submit(args):
    """Package benchmark results and submit via GitHub PR."""
    import os
    import json
    import glob
    import shutil
    import subprocess
    import tempfile
    from datetime import datetime
    from cogbenchv2 import __version__
    from cogbenchv2.config import RESULTS_DIR
    from cogbenchv2.evaluation.metrics import compute_metrics, compute_adversarial_gap
    from cogbenchv2.passages.processor import load_all_passages

    REPO = args.repo

    # ── Step 1: Find eval files ──────────────────────────────────────────────
    results_dir = args.results_dir or RESULTS_DIR
    eval_files = sorted(glob.glob(os.path.join(results_dir, "eval_*.json")))
    gen_files = sorted(glob.glob(os.path.join(results_dir, "gen_*.json")))

    if not eval_files:
        print("ERROR: No evaluation files found in", results_dir)
        print("Run 'cogbench run --model <model> --mode both' first.")
        sys.exit(1)

    # ── Step 2: Group by model ───────────────────────────────────────────────
    models = {}
    for f in eval_files:
        basename = os.path.basename(f).replace("eval_", "").replace(".json", "")
        if basename.endswith("_standard"):
            model_safe = basename[:-9]
            models.setdefault(model_safe, {})["standard"] = f
        elif basename.endswith("_adversarial"):
            model_safe = basename[:-12]
            models.setdefault(model_safe, {})["adversarial"] = f

    # Filter to complete models (both modes)
    complete = {k: v for k, v in models.items()
                if "standard" in v and "adversarial" in v}

    if not complete:
        print("ERROR: No models with both standard and adversarial evaluations found.")
        print("Run 'cogbench run --model <model> --mode both' to generate both modes.")
        sys.exit(1)

    # Select model
    if args.model:
        # Find matching model
        target = args.model.replace(":", "_").replace(".", "_").replace("/", "_")
        if target not in complete:
            print(f"ERROR: Model '{args.model}' not found. Available: {list(complete.keys())}")
            sys.exit(1)
        selected = {target: complete[target]}
    else:
        selected = complete

    print(f"{'='*60}")
    print(f"  COGBENCH SUBMISSION")
    print(f"{'='*60}")
    print(f"\n  Models to submit: {list(selected.keys())}")

    # ── Step 3: Build submission ─────────────────────────────────────────────
    passages = load_all_passages()
    submissions = {}

    for model_safe, files in selected.items():
        print(f"\n  Processing: {model_safe}")

        with open(files["standard"]) as f:
            std_evals = json.load(f).get("evaluations", [])
        with open(files["adversarial"]) as f:
            adv_evals = json.load(f).get("evaluations", [])

        std_metrics = compute_metrics(std_evals)
        adv_metrics = compute_metrics(adv_evals)
        gap_data = compute_adversarial_gap(std_metrics, adv_metrics)

        # Also read gen file metadata
        gen_std_path = os.path.join(results_dir, f"gen_{model_safe}_standard.json")
        gen_meta = {}
        if os.path.exists(gen_std_path):
            with open(gen_std_path) as f:
                gen_data = json.load(f)
            gen_meta = {
                "total_generations": len(gen_data.get("generations", [])),
                "errors": sum(1 for g in gen_data.get("generations", []) if g.get("error")),
            }

        submissions[model_safe] = {
            "standard_csr": std_metrics["prompt_level_strict"]["rate"],
            "adversarial_csr": adv_metrics["prompt_level_strict"]["rate"],
            "gap": gap_data["overall_gap"],
            "constraint_strict": std_metrics["constraint_level_strict"]["rate"],
            "constraint_loose": std_metrics["constraint_level_loose"]["rate"],
            "prompt_loose": std_metrics["prompt_level_loose"]["rate"],
            "n_standard": std_metrics["prompt_level_strict"].get("n", 0),
            "n_adversarial": adv_metrics["prompt_level_strict"].get("n", 0),
            **gen_meta,
        }

        print(f"    Standard CSR:    {submissions[model_safe]['standard_csr']:.1%}")
        print(f"    Adversarial CSR: {submissions[model_safe]['adversarial_csr']:.1%}")
        print(f"    Gap:             {submissions[model_safe]['gap']:.1%}")

    submission_data = {
        "cogbench_version": __version__,
        "submitted_at": datetime.now().isoformat(),
        "submitter": args.name or "anonymous",
        "models": submissions,
    }

    # ── Step 4: Save submission file ─────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submitter_slug = (args.name or "anonymous").lower().replace(" ", "_")
    submission_filename = f"submission_{submitter_slug}_{timestamp}.json"
    submission_path = os.path.join(results_dir, submission_filename)

    with open(submission_path, "w") as f:
        json.dump(submission_data, f, indent=2)
    print(f"\n  Submission saved: {submission_path}")

    # ── Step 5: Submit via GitHub ────────────────────────────────────────────
    gh = shutil.which("gh")
    if not gh:
        print(f"\n  'gh' CLI not found. To submit automatically:")
        print(f"    1. Install GitHub CLI: https://cli.github.com")
        print(f"    2. Run: gh auth login")
        print(f"    3. Re-run: cogbench submit")
        print(f"\n  Or manually open an issue at: https://github.com/{REPO}/issues/new")
        print(f"  and attach the file: {submission_path}")
        return

    # Check gh auth
    auth_check = subprocess.run([gh, "auth", "status"], capture_output=True, text=True)
    if auth_check.returncode != 0:
        print(f"\n  GitHub CLI not authenticated. Run: gh auth login")
        return

    # Build issue body
    model_list = "\n".join(
        f"| {m} | {d['standard_csr']:.1%} | {d['adversarial_csr']:.1%} | {d['gap']:.1%} | {d.get('n_standard', '?')} |"
        for m, d in submissions.items()
    )

    issue_body = f"""## CogBench Submission

**Submitter:** {args.name or 'anonymous'}
**CogBench Version:** {__version__}
**Date:** {datetime.now().strftime('%Y-%m-%d')}

### Results

| Model | Standard CSR | Adversarial CSR | Gap | N |
|-------|-------------|----------------|-----|---|
{model_list}

### Submission Data

```json
{json.dumps(submission_data, indent=2)}
```
"""

    model_names = ", ".join(selected.keys())
    issue_title = f"Submission: {model_names} ({args.name or 'anonymous'})"

    print(f"\n  Creating GitHub issue on {REPO}...")
    result = subprocess.run(
        [gh, "issue", "create",
         "--repo", REPO,
         "--title", issue_title,
         "--body", issue_body,
         "--label", "submission"],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        issue_url = result.stdout.strip()
        print(f"  Submitted! Issue: {issue_url}")
    else:
        # Label might not exist, try without it
        result = subprocess.run(
            [gh, "issue", "create",
             "--repo", REPO,
             "--title", issue_title,
             "--body", issue_body],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            issue_url = result.stdout.strip()
            print(f"  Submitted! Issue: {issue_url}")
        else:
            print(f"  Failed to create issue: {result.stderr}")
            print(f"  Submission file saved at: {submission_path}")
            print(f"  Please submit manually at: https://github.com/{REPO}/issues/new")


def cmd_info(args):
    """Print benchmark configuration and stats."""
    from cogbenchv2 import __version__
    from cogbenchv2.config import (
        PASSAGES_DIR, RESULTS_DIR, BLOOM_LEVELS, SUBJECTS,
        LOCAL_MODELS, API_MODELS, TOGETHER_MODELS,
    )
    from cogbenchv2.passages.processor import load_all_passages

    print(f"CogBench v{__version__}")
    print(f"{'='*50}")
    print(f"\nPaths:")
    print(f"  Passages: {PASSAGES_DIR}")
    print(f"  Results:  {RESULTS_DIR}")
    print(f"\nBenchmark Design:")
    print(f"  Bloom's Levels: {len(BLOOM_LEVELS)} ({', '.join(BLOOM_LEVELS.values())})")
    print(f"  Subjects:       {len(SUBJECTS)} ({', '.join(SUBJECTS)})")
    print(f"  Constraints:    28 (4 universal + 24 structural/semantic)")
    print(f"\nModels Configured:")
    print(f"  Local (Ollama): {len(LOCAL_MODELS)}")
    for m in LOCAL_MODELS:
        print(f"    - {m}")
    if API_MODELS:
        print(f"  API:            {len(API_MODELS)}")
        for m in API_MODELS:
            print(f"    - {m}")
    if TOGETHER_MODELS:
        print(f"  Together.ai:    {len(TOGETHER_MODELS)}")
        for m in TOGETHER_MODELS:
            print(f"    - {m}")

    passages = load_all_passages()
    print(f"\nPassage Data:")
    if passages:
        print(f"  Total passages: {len(passages)}")
        subjects_found = set(p["subject"] for p in passages)
        print(f"  Subjects found: {len(subjects_found)}")
    else:
        print("  No passages found. Run 'cogbench scrape' to fetch them.")


def main():
    parser = argparse.ArgumentParser(
        prog="cogbench",
        description="CogBench — Verifiable Cognitive Constraint Benchmark",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # cogbench run
    p_run = subparsers.add_parser("run", help="Generate questions and evaluate models")
    p_run.add_argument("--model", type=str, default=None, help="Single model to run")
    p_run.add_argument("--mode", type=str, default="both",
                       choices=["standard", "adversarial", "both"],
                       help="Generation mode (default: both)")
    p_run.add_argument("--local-all", action="store_true",
                       help="Run all local (Ollama) models")
    p_run.add_argument("--api-all", action="store_true",
                       help="Run all API models")
    p_run.add_argument("--together-all", action="store_true",
                       help="Run all Together.ai models")
    p_run.add_argument("--evaluate-only", action="store_true",
                       help="Skip generation, only evaluate existing results")
    p_run.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results")
    p_run.set_defaults(func=cmd_run)

    # cogbench evaluate
    p_eval = subparsers.add_parser("evaluate",
                                   help="Re-evaluate existing generation files")
    p_eval.set_defaults(func=cmd_evaluate)

    # cogbench leaderboard
    p_lb = subparsers.add_parser("leaderboard",
                                 help="Generate leaderboard data.json")
    p_lb.add_argument("--output", type=str, default=None,
                      help="Output path for data.json")
    p_lb.set_defaults(func=cmd_leaderboard)

    # cogbench scrape
    p_scrape = subparsers.add_parser("scrape",
                                     help="Scrape passages from OpenStax")
    p_scrape.add_argument("--subjects", nargs="+", default=None,
                          help="Subjects to scrape (default: all)")
    p_scrape.add_argument("--n-passages", type=int, default=None,
                          help="Passages per subject")
    p_scrape.add_argument("--skip-process", action="store_true",
                          help="Skip NLP key concept extraction")
    p_scrape.set_defaults(func=cmd_scrape)

    # cogbench submit
    p_submit = subparsers.add_parser("submit",
                                     help="Submit results to the CogBench leaderboard")
    p_submit.add_argument("--model", type=str, default=None,
                          help="Specific model to submit (default: all completed)")
    p_submit.add_argument("--name", type=str, default=None,
                          help="Your name or team name")
    p_submit.add_argument("--repo", type=str, default="cogbench/cogbench",
                          help="GitHub repo for submission (default: cogbench/cogbench)")
    p_submit.add_argument("--results-dir", type=str, default=None,
                          help="Results directory to read from")
    p_submit.set_defaults(func=cmd_submit)

    # cogbench info
    p_info = subparsers.add_parser("info",
                                   help="Show benchmark config and stats")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
