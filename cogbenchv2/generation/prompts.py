"""Prompt templates for standard and adversarial question generation."""

from cogbenchv2.config import BLOOM_LEVELS, BLOOM_DEFINITIONS, BLOOM_VERBS, ADVERSARIAL_PAIRINGS


# ─── Standard prompt ──────────────────────────────────────────────────────────

STANDARD_TEMPLATE = """You are generating an educational question from a textbook passage.

PASSAGE:
{passage_text}

TASK: Generate a {level_name}-level question (Bloom's Taxonomy) about this passage.

At the {level_name} level, students should be able to {definition}.

{constraint_hints}

Respond in this EXACT format (nothing else):
QUESTION: [your question]
ANSWER: [a reference answer]"""


# ─── Adversarial prompt ──────────────────────────────────────────────────────

ADVERSARIAL_TEMPLATE = """You are generating an educational question from a textbook passage.

PASSAGE:
{passage_text}

TASK: Generate a {target_level_name}-level question (Bloom's Taxonomy), but you MUST use vocabulary from the {vocab_level_name} level.

The cognitive demand of your question must be {target_level_name} (students should {target_definition}), even though the wording uses {vocab_level_name}-level verbs like: {verb_list}.

IMPORTANT: Do NOT use verbs like {forbidden_verbs}. Instead, use verbs like: {required_verbs}.

{constraint_hints}

Respond in this EXACT format (nothing else):
QUESTION: [your question]
ANSWER: [a reference answer]"""


# ─── Constraint hints per level (human-readable, embedded in prompts) ────────

CONSTRAINT_HINTS = {
    1: """Your question must:
- Ask for a single, specific fact from the passage
- Have a short answer (under 20 words) that can be found directly in the text
- Target only one concept""",

    2: """Your question must:
- Ask for an explanation or interpretation, not just a bare fact
- Use words like "how" or "why" or "explain"
- The answer should paraphrase the passage, not copy it word-for-word
- The answer must still be supported by the passage content""",

    3: """Your question must:
- Present a NEW scenario or problem not described in the passage
- Require using a method, formula, or principle FROM the passage to solve it
- Have an answer with a specific result (a number, outcome, or concrete steps)""",

    4: """Your question must:
- Reference at least 2 different concepts from the passage
- Ask about the relationship between these concepts (compare, contrast, connect)
- The answer must address all concepts mentioned in the question""",

    5: """Your question must:
- Present a claim or position for the student to evaluate
- Ask for evidence-based reasoning (use "justify", "based on", "why or why not")
- The answer should contain argumentation with contrasting viewpoints""",

    6: """Your question must:
- Ask the student to design, create, or propose something original
- Include at least 2 specific requirements or constraints
- The answer should be substantial (more than 50 words) and go beyond what's in the passage""",
}


def build_standard_prompt(passage_text: str, level: int) -> str:
    """Build a standard generation prompt."""
    return STANDARD_TEMPLATE.format(
        passage_text=passage_text,
        level_name=BLOOM_LEVELS[level],
        definition=BLOOM_DEFINITIONS[level],
        constraint_hints=CONSTRAINT_HINTS.get(level, ""),
    )


def build_adversarial_prompt(passage_text: str, target_level: int,
                              vocab_level: int) -> str:
    """Build an adversarial generation prompt.

    Args:
        passage_text: The source passage
        target_level: The cognitive level the question should actually be at
        vocab_level: The level whose vocabulary must be used
    """
    required_verbs = ", ".join(BLOOM_VERBS[vocab_level][:6])
    forbidden_verbs = ", ".join(BLOOM_VERBS[target_level][:6])

    return ADVERSARIAL_TEMPLATE.format(
        passage_text=passage_text,
        target_level_name=BLOOM_LEVELS[target_level],
        vocab_level_name=BLOOM_LEVELS[vocab_level],
        target_definition=BLOOM_DEFINITIONS[target_level],
        verb_list=", ".join(BLOOM_VERBS[vocab_level][:8]),
        forbidden_verbs=forbidden_verbs,
        required_verbs=required_verbs,
        constraint_hints=CONSTRAINT_HINTS.get(target_level, ""),
    )


def get_adversarial_pairing(target_level: int):
    """Get the vocab_level for a given target_level in adversarial mode.

    Returns vocab_level or None if no pairing exists.
    """
    for tgt, vocab in ADVERSARIAL_PAIRINGS:
        if tgt == target_level:
            return vocab
    return None
