# CogBench

**Verifiable Cognitive Constraint Benchmark for LLM Question Generation**

CogBench evaluates whether large language models can generate educationally appropriate questions at specific Bloom's Taxonomy cognitive levels while satisfying 28 deterministic constraints.

## Features

- **6 Bloom's Taxonomy Levels**: Remember, Understand, Apply, Analyze, Evaluate, Create
- **8 Academic Subjects**: Biology, Chemistry, Physics, Mathematics, Psychology, Economics, History, Computer Science
- **28 Deterministic Constraints**: No LLM-as-judge — every constraint is verifiable and reproducible
- **Adversarial Mode**: Tests robustness by using mismatched cognitive-level vocabulary
- **120 Passages**: From OpenStax open-access textbooks

## Installation

```bash
pip install cogbench
```

For NLP features (key concept extraction with spaCy):
```bash
pip install cogbench[nlp]
python -m spacy download en_core_web_sm
```

## Quick Start

```bash
# Show benchmark configuration
cogbench info

# Run benchmark for a single model (requires Ollama)
cogbench run --model qwen2.5:14b --mode standard

# Run all local models in both modes
cogbench run --local-all --mode both

# Re-evaluate existing results with updated constraints
cogbench evaluate

# Generate leaderboard data
cogbench leaderboard
```

## CLI Commands

| Command | Description |
|---|---|
| `cogbench info` | Show benchmark config, models, and passage stats |
| `cogbench run` | Generate questions and evaluate models |
| `cogbench evaluate` | Re-evaluate existing generation files |
| `cogbench leaderboard` | Populate leaderboard data.json from results |
| `cogbench scrape` | Scrape passages from OpenStax textbooks |
| `cogbench submit` | Submit results to the CogBench leaderboard via GitHub |

## Submitting Results

After running the benchmark, submit your results to the public leaderboard:

```bash
# Submit all completed models
cogbench submit --name "Your Name"

# Submit a specific model
cogbench submit --model qwen2.5:14b --name "Your Name"
```

This creates a GitHub issue with your results. The maintainers will review and add them to the leaderboard. Requires the [GitHub CLI](https://cli.github.com) (`gh`).

## Python API

```python
from cogbenchv2.passages.processor import load_all_passages
from cogbenchv2.evaluation.evaluate import evaluate_generations
from cogbenchv2.evaluation.metrics import compute_metrics

# Load the 120 bundled passages
passages = load_all_passages()

# Evaluate a generation file
evaluations = evaluate_generations("gen_qwen2_5_14b_standard.json", passages)

# Compute metrics
metrics = compute_metrics(evaluations)
print(f"Prompt-level strict: {metrics['prompt_level_strict']['rate']:.1%}")
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `COGBENCH_RESULTS_DIR` | `./data/results` | Where benchmark results are saved |
| `OPENAI_API_KEY` | — | For OpenAI API models |
| `GOOGLE_API_KEY` | — | For Google Gemini models |
| `TOGETHER_API_KEY` | — | For Together.ai models |

## License

Apache 2.0
