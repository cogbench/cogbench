"""CogBench configuration — all constants, verb lists, and thresholds."""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Bundled passage data lives inside the package (cogbenchv2/data/passages/)
PASSAGES_DIR = os.path.join(PACKAGE_DIR, "data", "passages")

# User-writable output — defaults to cwd/cogbench_results, override with env var
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.environ.get(
    "COGBENCH_RESULTS_DIR",
    os.path.join(DATA_DIR, "results"),
)

# ─── Bloom's Taxonomy ────────────────────────────────────────────────────────

BLOOM_LEVELS = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create",
}

BLOOM_DEFINITIONS = {
    1: "recall facts and basic concepts from the passage",
    2: "explain ideas or concepts in their own words",
    3: "use information in new situations or solve problems",
    4: "draw connections among ideas, compare and contrast",
    5: "justify a stand or decision, critique or evaluate claims",
    6: "produce new or original work, design or propose something",
}

# ─── Bloom's Verb Lists (used for vocabulary constraints) ────────────────────
# Each level has canonical verbs. In standard mode, questions must use verbs
# from their target level. In adversarial mode, they use verbs from a
# different level.

BLOOM_VERBS = {
    1: [  # Remember
        "define", "list", "recall", "identify", "name", "state",
        "what is", "who", "when", "where", "which", "label",
        "recognize", "match", "select", "memorize",
    ],
    2: [  # Understand
        "explain", "describe", "summarize", "interpret", "paraphrase",
        "illustrate", "discuss", "classify", "compare", "distinguish",
        "infer", "predict", "restate", "translate",
    ],
    3: [  # Apply
        "calculate", "solve", "apply", "demonstrate", "implement",
        "use", "compute", "determine", "execute", "practice",
        "operate", "sketch", "modify",
    ],
    4: [  # Analyze
        "compare", "contrast", "examine", "differentiate", "categorize",
        "distinguish", "relate", "analyze", "organize", "deconstruct",
        "attribute", "outline", "investigate",
    ],
    5: [  # Evaluate
        "assess", "critique", "judge", "justify", "argue", "defend",
        "evaluate", "recommend", "to what extent", "appraise",
        "prioritize", "rate", "support", "conclude",
    ],
    6: [  # Create
        "design", "develop", "propose", "construct", "formulate",
        "synthesize", "create", "devise", "plan", "compose",
        "invent", "produce", "generate", "elaborate",
    ],
}

# ─── Subjects ─────────────────────────────────────────────────────────────────

SUBJECTS = [
    "biology", "chemistry", "physics", "mathematics",
    "psychology", "economics", "history", "computer_science",
]

# OpenStax textbook slugs per subject (used by scraper)
OPENSTAX_BOOKS = {
    "biology": "biology-2e",
    "chemistry": "chemistry-2e",
    "physics": "college-physics-2e",
    "mathematics": "calculus-volume-1",
    "psychology": "psychology-2e",
    "economics": "principles-economics-3e",
    "history": "us-history",
    "computer_science": "introduction-computer-science",
}

PASSAGES_PER_SUBJECT = 15  # Target: ~120 total passages

# ─── Generation Parameters ────────────────────────────────────────────────────

TEMPERATURE = 0.7
MAX_TOKENS = 512  # Need room for question + answer
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 120

# ─── Models ───────────────────────────────────────────────────────────────────

LOCAL_MODELS = {
    "qwen2.5:14b": {"backend": "ollama", "name": "Qwen 2.5 14B", "params": "14B"},
    "llama3.1:8b": {"backend": "ollama", "name": "Llama 3.1 8B", "params": "8B"},
    "gemma3:12b": {"backend": "ollama", "name": "Gemma 3 12B", "params": "12B"},
    "phi4:14b": {"backend": "ollama", "name": "Phi-4 14B", "params": "14B"},
    "deepseek-r1:14b": {"backend": "ollama", "name": "DeepSeek-R1 14B", "params": "14B"},
    "mistral-nemo:12b": {"backend": "ollama", "name": "Mistral Nemo 12B", "params": "12B"},
    "gemma2:9b": {"backend": "ollama", "name": "Gemma 2 9B", "params": "9B"},
}

API_MODELS = {
    "gpt-4o": {"backend": "openai", "name": "GPT-4o", "params": "~200B"},
    "claude-3-5-sonnet-20241022": {"backend": "anthropic", "name": "Claude 3.5 Sonnet", "params": "~70B"},
    "gemini-2.5-flash": {"backend": "google", "name": "Gemini 2.5 Flash", "params": "frontier"},
}

TOGETHER_MODELS = {
    "deepseek-ai/DeepSeek-V3.1": {"backend": "together", "name": "DeepSeek-V3.1", "params": "671B MoE"},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"backend": "together", "name": "Llama 3.3 70B", "params": "70B"},
}

ALL_MODELS = {**LOCAL_MODELS, **API_MODELS, **TOGETHER_MODELS}

# API keys from environment
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ─── Constraint Thresholds ────────────────────────────────────────────────────
# Tier 1 (structural)

# Universal
U_MIN_WORDS = 10
U_MAX_WORDS = 150
U_MIN_PASSAGE_TERMS = 2  # Min key terms from passage in question
U_MAX_WORD_REPEAT = 3    # Max times any word can repeat

# L1 Remember
R_MAX_CONCEPTS = 1       # Only 1 key concept in question
R_MAX_ANSWER_WORDS = 20  # Short factual answer
R_PASSAGE_OVERLAP = 0.60 # 60% of answer words findable in passage

# L2 Understand
D_MAX_PASSAGE_OVERLAP = 0.70  # Answer must NOT be copy-pasted (<70% stemmed word overlap)

# L3 Apply
# (no structural thresholds beyond verb check)

# L4 Analyze
A_MIN_CONCEPTS = 2  # At least 2 key concepts from passage

# L5 Evaluate
# (uses verb + marker checks)

# L6 Create
C_MIN_SPEC_MARKERS = 2   # At least 2 specification markers
C_MIN_ANSWER_WORDS = 50  # Creation requires substantial output

# Tier 2 (NLI / semantic)
NLI_MODEL_NAME = "roberta-large-mnli"
NLI_ENTAILMENT_THRESHOLD = 0.65   # D3: passage entails answer
NLI_NOVELTY_THRESHOLD = 0.55      # P2, C2: passage does NOT entail scenario
NLI_DEVICE = 1                    # GPU device for NLI model

# ─── Adversarial Pairings ────────────────────────────────────────────────────
# (target_level, vocab_from_level)

ADVERSARIAL_PAIRINGS = [
    (2, 1),  # Generate Understand using Remember verbs
    (3, 2),  # Generate Apply using Understand verbs
    (4, 1),  # Generate Analyze using Remember verbs
    (5, 2),  # Generate Evaluate using Understand verbs
    (6, 3),  # Generate Create using Apply verbs
    (1, 4),  # Generate Remember using Analyze verbs
]

# ─── Evaluation Metrics ──────────────────────────────────────────────────────

SEED = 42
BOOTSTRAP_N = 1000       # Bootstrap samples for confidence intervals
CONFIDENCE_LEVEL = 0.95  # 95% CI
