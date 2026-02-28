"""Process scraped passages — extract key concepts and methods/principles.

Uses spaCy NLP for noun phrase extraction, plus heuristic rules
for identifying methods, formulas, and principles.
"""

import os
import re
import json
from collections import Counter
from typing import Optional

from cogbenchv2.config import PASSAGES_DIR, SUBJECTS


def extract_key_concepts(text: str, top_n: int = 8) -> list:
    """Extract key concepts from passage text using noun phrase frequency.

    Falls back to simple noun extraction if spaCy is not available.

    Args:
        text: Passage text
        top_n: Number of top concepts to return

    Returns:
        List of key concept strings
    """
    try:
        import spacy
        nlp = _get_spacy_model()
        return _extract_with_spacy(nlp, text, top_n)
    except ImportError:
        return _extract_simple(text, top_n)


def _get_spacy_model():
    """Load spaCy model (cached)."""
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Download if not available
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def _extract_with_spacy(nlp, text: str, top_n: int) -> list:
    """Extract noun phrases using spaCy."""
    doc = nlp(text[:10000])  # Limit for performance

    # Collect noun phrases
    phrases = []
    for chunk in doc.noun_chunks:
        # Clean up: remove determiners, pronouns
        phrase = chunk.text.strip().lower()
        # Remove leading articles
        phrase = re.sub(r'^(the|a|an|this|that|these|those|its|their|our)\s+', '', phrase)
        if len(phrase) > 2 and len(phrase.split()) <= 4:
            phrases.append(phrase)

    # Also get named entities
    for ent in doc.ents:
        if ent.label_ in {"ORG", "PERSON", "GPE", "EVENT", "WORK_OF_ART",
                           "LAW", "PRODUCT", "NORP"}:
            continue  # Skip non-concept entities
        phrases.append(ent.text.lower())

    # Rank by frequency
    counter = Counter(phrases)

    # Filter out common stop-phrases and stop words
    stop_phrases = {"it", "they", "we", "you", "one", "way", "time", "part",
                    "example", "figure", "table", "chapter", "section", "page",
                    "result", "case", "number", "type", "form", "end",
                    "process", "system", "use", "order", "point",
                    # Stop words that spaCy sometimes includes as noun phrases
                    "that", "this", "which", "what", "who", "how", "all",
                    "these", "those", "such", "each", "some", "any", "many",
                    "both", "other", "than", "more", "most", "much", "very",
                    "also", "only", "just", "even", "still", "well",
                    }

    ranked = [(phrase, count) for phrase, count in counter.most_common(top_n * 3)
              if phrase not in stop_phrases and len(phrase) > 2]

    # Deduplicate singular/plural forms — keep the more frequent form
    deduped = []
    seen_stems = set()
    for phrase, count in ranked:
        # Simple dedup: check if singular/plural already seen
        stem = phrase.rstrip("s").rstrip("e")  # rough singularization
        if stem not in seen_stems and phrase not in seen_stems:
            seen_stems.add(stem)
            seen_stems.add(phrase)
            deduped.append(phrase)

    return deduped[:top_n]


def _extract_simple(text: str, top_n: int) -> list:
    """Simple fallback: extract capitalized multi-word terms and frequent nouns."""
    # Find capitalized terms (likely proper nouns / technical terms)
    cap_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    cap_terms = [t.lower() for t in cap_terms]

    # Find words that appear often (likely key concepts)
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    stop = {"this", "that", "with", "from", "have", "been", "were", "they",
            "their", "which", "would", "could", "about", "there", "these",
            "other", "also", "more", "than", "into", "some", "when",
            "only", "each", "such", "most", "very", "much", "many",
            "just", "over", "between", "through", "same", "after", "before"}
    content_words = [w for w in words if w not in stop]

    counter = Counter(content_words)
    frequent = [w for w, c in counter.most_common(top_n * 2) if c >= 2]

    # Combine: capitalized terms first, then frequent terms
    seen = set()
    result = []
    for term in cap_terms + frequent:
        if term not in seen:
            seen.add(term)
            result.append(term)
        if len(result) >= top_n:
            break

    return result


def extract_methods_principles(text: str) -> list:
    """Extract methods, formulas, principles, and laws from passage text.

    Looks for patterns like "X's law", "the principle of X", "the X method",
    "formula for X", etc.
    """
    methods = []

    # Named laws/principles/theories
    patterns = [
        r"([A-Z][a-z]+(?:'s)?\s+(?:law|principle|theory|theorem|equation|rule|effect|model|hypothesis|method|paradox|constant))",
        r"(?:the\s+)?(?:law|principle|theory|theorem)\s+of\s+([\w\s]+?)(?:\.|,|\s+is|\s+states)",
        r"(?:the\s+)?([\w\s]+?)\s+(?:formula|equation|method|technique|algorithm|procedure|process)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            cleaned = m.strip().lower()
            if len(cleaned) > 3 and len(cleaned.split()) <= 5:
                methods.append(cleaned)

    # Deduplicate
    seen = set()
    unique = []
    for m in methods:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    return unique[:6]  # Max 6 methods per passage


def process_passage(passage: dict) -> dict:
    """Process a single passage — add key_concepts and methods_principles."""
    text = passage.get("text", "")
    if not text:
        return passage

    passage["key_concepts"] = extract_key_concepts(text)
    passage["methods_principles"] = extract_methods_principles(text)

    return passage


def process_all_passages():
    """Process all scraped passages — add NLP-extracted metadata."""
    total_processed = 0

    for subject in SUBJECTS:
        subject_dir = os.path.join(PASSAGES_DIR, subject)
        passages_file = os.path.join(subject_dir, "passages.json")

        if not os.path.exists(passages_file):
            print(f"  Skipping {subject} — no passages.json")
            continue

        with open(passages_file) as f:
            passages = json.load(f)

        print(f"\n  Processing {subject}: {len(passages)} passages")

        for i, p in enumerate(passages):
            passages[i] = process_passage(p)
            concepts = passages[i]["key_concepts"]
            methods = passages[i]["methods_principles"]
            print(f"    {p['passage_id']}: {len(concepts)} concepts, {len(methods)} methods")

        # Save back
        with open(passages_file, "w") as f:
            json.dump(passages, f, indent=2)

        total_processed += len(passages)

    print(f"\n  Total processed: {total_processed}")
    return total_processed


def load_all_passages() -> list:
    """Load all processed passages from disk.

    Returns flat list of passage dicts.
    """
    all_passages = []

    for subject in SUBJECTS:
        subject_dir = os.path.join(PASSAGES_DIR, subject)
        passages_file = os.path.join(subject_dir, "passages.json")

        if not os.path.exists(passages_file):
            continue

        with open(passages_file) as f:
            passages = json.load(f)
            all_passages.extend(passages)

    return all_passages
