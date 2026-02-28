"""Scrape textbook passages from OpenStax.

OpenStax textbooks are free, CC-BY licensed, and have clean HTML.
We use the archive API for TOC and direct page scraping for content.
"""

import os
import re
import json
import time
import random
import requests
from bs4 import BeautifulSoup
from typing import Optional

from cogbenchv2.config import (
    OPENSTAX_BOOKS, SUBJECTS, PASSAGES_PER_SUBJECT, PASSAGES_DIR,
)


OPENSTAX_BASE = "https://openstax.org"
OPENSTAX_PAGE_URL = "https://openstax.org/books/{book}/pages/{page}"

# Book UUIDs (from OpenStax CMS API)
BOOK_UUIDS = {
    "biology-2e": "8d50a0af-948b-4204-a71d-4826cba765b8",
    "chemistry-2e": "7fccc9cf-9b71-44f6-800b-f9457fd64335",
    "college-physics-2e": "a31df062-930a-4f46-8953-605711e6d204",
    "calculus-volume-1": "8b89d172-2927-466f-8661-01abc7ccdba4",
    "psychology-2e": "06aba565-9432-40f6-97ee-b8a361f118a8",
    "principles-economics-3e": "4c34671f-b057-4918-8796-38ca1b2f4151",
    "us-history": "a7ba2fb8-8925-4987-b182-5f4429d48daa",
    "introduction-computer-science": "4f06ce31-1d04-410f-8328-ac4e02c4f217",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0",
    "Accept": "text/html,application/xhtml+xml,application/json",
}
REQUEST_DELAY = 2.0

# Slug patterns to skip (non-content pages)
SKIP_PATTERNS = [
    "key-terms", "chapter-summary", "review-questions", "critical-thinking",
    "visual-connection", "introduction", "preface", "appendix", "index",
    "glossary", "further-research", "references", "bibliography",
    "multiple-choice", "short-answer", "free-response", "practice",
    "conceptual-questions", "problems", "additional-problems",
    "challenge-problems", "check-understanding", "section-summary",
]


def _get_archive_info():
    """Get the current archive URL and book versions from OpenStax."""
    resp = requests.get(f"{OPENSTAX_BASE}/rex/release.json",
                        headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data["archiveUrl"], data["books"]


def get_table_of_contents(book_slug: str) -> list:
    """Get content sections for an OpenStax book via the archive API.

    Returns list of {"title": str, "slug": str} for actual content sections only.
    """
    uuid = BOOK_UUIDS.get(book_slug)
    if not uuid:
        print(f"  No UUID configured for '{book_slug}'")
        return []

    try:
        archive_url, books = _get_archive_info()
        version = books.get(uuid, {}).get("defaultVersion", "")
        if not version:
            print(f"  No version found for {book_slug}")
            return []

        toc_url = f"{OPENSTAX_BASE}{archive_url}/contents/{uuid}@{version}.json"
        resp = requests.get(toc_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Failed to get TOC for {book_slug}: {e}")
        return []

    tree = data.get("tree", {})

    # Walk the tree to collect leaf sections
    sections = []

    def _walk(node):
        slug = node.get("slug", "")
        title = node.get("title", "")
        children = node.get("contents", [])

        if not children and slug:
            # Skip non-content pages
            if any(skip in slug for skip in SKIP_PATTERNS):
                return
            # Must have a numbered section pattern (e.g., 4-3-xxx)
            if re.match(r"^\d+-\d+", slug):
                clean_title = re.sub(r"<[^>]+>", "", title).strip()
                sections.append({"title": clean_title, "slug": slug})

        for child in children:
            _walk(child)

    _walk(tree)
    return sections


def scrape_page(book_slug: str, page_slug: str) -> Optional[str]:
    """Scrape a single OpenStax page and extract the main content text."""
    url = OPENSTAX_PAGE_URL.format(book=book_slug, page=page_slug)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"    Failed to fetch {page_slug}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Content is in <div data-type="page">
    content = soup.find("div", {"data-type": "page"})
    if not content:
        content = soup.find("main")
    if not content:
        return None

    # Remove non-text elements
    for tag in content.find_all(["figure", "table", "footer", "nav",
                                  "script", "style", "aside", "svg", "math"]):
        tag.decompose()

    # Remove exercise/problem/review sections
    for tag in content.find_all(class_=re.compile(
            r"exercise|problem|solution|review|key-terms|glossary|os-teacher")):
        tag.decompose()

    # Remove learning objectives sections
    for tag in content.find_all(class_=re.compile(r"learning-objectives|abstract")):
        tag.decompose()

    # Extract text from paragraphs
    paragraphs = content.find_all("p")
    text_parts = []
    for p in paragraphs:
        text = p.get_text(separator=" ", strip=True)
        # Skip very short fragments, figure references, and equation labels
        if text and len(text) > 40:
            # Skip "Figure X.Y" references and similar
            if re.match(r"^(Figure|Table|Equation)\s+\d", text):
                continue
            text_parts.append(text)

    full_text = " ".join(text_parts)
    full_text = re.sub(r"\s+", " ", full_text).strip()

    # Clean up unicode artifacts
    full_text = full_text.replace("\u2019", "'").replace("\u2018", "'")
    full_text = full_text.replace("\u201c", '"').replace("\u201d", '"')
    full_text = full_text.replace("\u2013", "-").replace("\u2014", "--")

    return full_text if len(full_text) > 150 else None


def extract_section(full_text: str, target_words: int = 500,
                    min_words: int = 300, max_words: int = 800) -> Optional[str]:
    """Extract a passage of target length from the full page text."""
    words = full_text.split()
    n = len(words)

    if n < min_words:
        return None
    if n <= max_words:
        return full_text

    # Take the first target_words
    passage = " ".join(words[:target_words])

    # End at a sentence boundary
    last_period = passage.rfind(".")
    if last_period > len(passage) * 0.6:
        passage = passage[:last_period + 1]

    return passage


def scrape_subject(subject: str, n_passages: int = PASSAGES_PER_SUBJECT,
                   save: bool = True) -> list:
    """Scrape passages for one subject from OpenStax.

    Selects sections spread across the book (not just the first N).
    """
    book_slug = OPENSTAX_BOOKS.get(subject)
    if not book_slug:
        print(f"  No OpenStax book configured for '{subject}'")
        return []

    print(f"\n  Scraping {subject} ({book_slug})...")
    all_sections = get_table_of_contents(book_slug)
    print(f"  Found {len(all_sections)} content sections")

    if not all_sections:
        print(f"  Warning: No sections found for {subject}")
        return []

    # Spread selections across the book (every Nth section)
    if len(all_sections) > n_passages:
        step = len(all_sections) / n_passages
        indices = [int(i * step) for i in range(n_passages)]
        selected = [all_sections[i] for i in indices]
    else:
        selected = all_sections[:n_passages]

    passages = []
    for page in selected:
        if len(passages) >= n_passages:
            break

        slug = page["slug"]
        title = page["title"]

        text = scrape_page(book_slug, slug)
        if not text:
            continue

        section = extract_section(text)
        if not section:
            continue

        passage_id = f"{subject[:3]}_{len(passages) + 1:03d}"
        passage = {
            "passage_id": passage_id,
            "subject": subject,
            "source": f"openstax_{book_slug}",
            "section": title,
            "text": section,
            "key_concepts": [],
            "methods_principles": [],
            "url": OPENSTAX_PAGE_URL.format(book=book_slug, page=slug),
            "word_count": len(section.split()),
        }
        passages.append(passage)
        print(f"    [{len(passages)}/{n_passages}] {title} ({passage['word_count']} words)")

        time.sleep(REQUEST_DELAY)

    # Save
    if save and passages:
        subject_dir = os.path.join(PASSAGES_DIR, subject)
        os.makedirs(subject_dir, exist_ok=True)
        save_path = os.path.join(subject_dir, "passages.json")
        with open(save_path, "w") as f:
            json.dump(passages, f, indent=2)
        print(f"  Saved {len(passages)} passages to {save_path}")

    return passages


def scrape_all(n_passages: int = PASSAGES_PER_SUBJECT) -> dict:
    """Scrape passages for all subjects."""
    all_passages = {}
    for subject in SUBJECTS:
        passages = scrape_subject(subject, n_passages)
        all_passages[subject] = passages
        time.sleep(REQUEST_DELAY * 2)

    total = sum(len(v) for v in all_passages.values())
    print(f"\n  Total passages scraped: {total}")
    return all_passages
