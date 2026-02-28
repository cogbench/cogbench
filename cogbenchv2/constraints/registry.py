"""Constraint registry — maps each Bloom's level to its list of constraints.

Central place to get "what constraints apply to level X in mode Y?"
"""

from cogbenchv2.constraints.universal import UNIVERSAL_CONSTRAINTS
from cogbenchv2.constraints.structural import (
    REMEMBER_CONSTRAINTS, UNDERSTAND_CONSTRAINTS, APPLY_CONSTRAINTS,
    ANALYZE_CONSTRAINTS, EVALUATE_CONSTRAINTS, CREATE_CONSTRAINTS,
)
from cogbenchv2.constraints.semantic import (
    UnderstandSupported, ApplyNewScenario, CreateNovel,
)
from cogbenchv2.constraints.adversarial import ADVERSARIAL_CONSTRAINTS


# Level-specific constraints (structural + semantic combined)
LEVEL_CONSTRAINTS = {
    1: REMEMBER_CONSTRAINTS,   # R1, R2, R3, R4
    2: UNDERSTAND_CONSTRAINTS + [UnderstandSupported()],  # D1, D2, D4 + D3
    3: APPLY_CONSTRAINTS + [ApplyNewScenario()],          # P1, P3, P4 + P2
    4: ANALYZE_CONSTRAINTS,    # A1, A2, A3, A4
    5: EVALUATE_CONSTRAINTS,   # E1, E2, E3, E4
    6: CREATE_CONSTRAINTS + [CreateNovel()],              # C1, C3, C4 + C2
}


def get_constraints(level: int, mode: str = "standard"):
    """Get all constraints for a given level and mode.

    Args:
        level: Bloom's level (1-6)
        mode: "standard" or "adversarial"

    Returns:
        List of Constraint objects to check
    """
    constraints = list(UNIVERSAL_CONSTRAINTS)  # Copy — always apply universals

    level_specific = LEVEL_CONSTRAINTS.get(level, [])

    if mode == "standard":
        # Standard: all constraints including vocabulary
        constraints.extend(level_specific)
    elif mode == "adversarial":
        # Adversarial: structural constraints MINUS the vocabulary constraint,
        # PLUS adversarial vocabulary constraints
        for c in level_specific:
            # Skip the level's own vocabulary constraint (X1 where X is the level prefix)
            if c.constraint_id.endswith("1") and c.tier == "structural":
                # This is the vocabulary constraint — replace with adversarial
                continue
            constraints.append(c)
        constraints.extend(ADVERSARIAL_CONSTRAINTS)

    return constraints


def get_all_constraint_ids(level: int, mode: str = "standard"):
    """Get all constraint IDs for a given level and mode."""
    return [c.constraint_id for c in get_constraints(level, mode)]


def get_constraint_count(level: int, mode: str = "standard"):
    """How many constraints apply to this level/mode?"""
    return len(get_constraints(level, mode))


def describe_constraints(level: int, mode: str = "standard"):
    """Human-readable description of all constraints for a level.

    Used in generation prompts to tell the LLM what rules to follow.
    """
    descriptions = []
    for c in get_constraints(level, mode):
        if c.tier == "universal":
            continue  # Don't tell the model about universal constraints
        descriptions.append(f"- {c.constraint_name}: {c.__doc__.split(chr(10))[0].strip()}")
    return "\n".join(descriptions)
