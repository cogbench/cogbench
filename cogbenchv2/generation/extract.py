"""Parse question + answer from LLM output.

LLMs are instructed to respond in the format:
    QUESTION: [question text]
    ANSWER: [answer text]

This module handles various deviations from the expected format.
"""

import re
from typing import Optional, Tuple


def extract_qa(raw_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract question and answer from LLM response.

    Args:
        raw_text: Raw LLM output

    Returns:
        (question, answer) tuple. Either can be None if extraction fails.
    """
    if not raw_text or not raw_text.strip():
        return None, None

    text = raw_text.strip()

    # Strategy 1: Look for QUESTION: and ANSWER: markers
    q_match = re.search(r'QUESTION:\s*(.+?)(?=ANSWER:|$)', text,
                        re.IGNORECASE | re.DOTALL)
    a_match = re.search(r'ANSWER:\s*(.+)', text, re.IGNORECASE | re.DOTALL)

    if q_match and a_match:
        question = _clean_text(q_match.group(1))
        answer = _clean_text(a_match.group(1))
        return question, answer

    # Strategy 2: Look for Q: and A: markers
    q_match = re.search(r'(?:^|\n)\s*Q:\s*(.+?)(?=\nA:|$)', text,
                        re.IGNORECASE | re.DOTALL)
    a_match = re.search(r'(?:^|\n)\s*A:\s*(.+)', text, re.IGNORECASE | re.DOTALL)

    if q_match and a_match:
        question = _clean_text(q_match.group(1))
        answer = _clean_text(a_match.group(1))
        return question, answer

    # Strategy 3: Look for "?" to split question from answer
    q_end = text.rfind("?")
    if q_end > 0:
        question = _clean_text(text[:q_end + 1])
        answer = _clean_text(text[q_end + 1:])
        if answer:
            return question, answer
        # If no answer after the question mark, question is the whole thing
        return question, None

    # Strategy 4: First line is question, rest is answer
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 2:
        return _clean_text(lines[0]), _clean_text("\n".join(lines[1:]))

    # Last resort: everything is the question, no answer
    return _clean_text(text), None


def _clean_text(text: str) -> Optional[str]:
    """Clean extracted text — remove quotes, numbering, extra whitespace."""
    if not text:
        return None

    text = text.strip()

    # Remove surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Remove leading numbering (1., 1), etc.)
    text = re.sub(r'^\d+[\.\)]\s*', '', text)

    # Remove "Question:" or "Answer:" prefix if still present
    text = re.sub(r'^(?:Question|Answer|Q|A)\s*:\s*', '', text, flags=re.IGNORECASE)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text if text else None
