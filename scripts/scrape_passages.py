"""Step 1: Scrape textbook passages from OpenStax.

Usage:
    python scripts/scrape_passages.py
    python scripts/scrape_passages.py --subjects biology chemistry
    python scripts/scrape_passages.py --n-passages 5 --subjects biology  # Quick test
"""

import sys
import os
import argparse
from cogbenchv2.passages.scraper import scrape_all, scrape_subject
from cogbenchv2.passages.processor import process_all_passages
from cogbenchv2.config import SUBJECTS, PASSAGES_PER_SUBJECT


def main():
    parser = argparse.ArgumentParser(description="Scrape OpenStax passages")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help=f"Subjects to scrape (default: all {len(SUBJECTS)})")
    parser.add_argument("--n-passages", type=int, default=PASSAGES_PER_SUBJECT,
                        help=f"Passages per subject (default: {PASSAGES_PER_SUBJECT})")
    parser.add_argument("--skip-process", action="store_true",
                        help="Skip NLP processing (key concept extraction)")
    args = parser.parse_args()

    subjects = args.subjects or SUBJECTS

    print(f"Scraping {len(subjects)} subjects, {args.n_passages} passages each")
    print(f"Subjects: {subjects}\n")

    for subject in subjects:
        scrape_subject(subject, args.n_passages)

    if not args.skip_process:
        print("\n\nProcessing passages (extracting key concepts)...")
        process_all_passages()

    print("\nDone!")


if __name__ == "__main__":
    main()
