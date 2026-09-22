"""Build the books-first website catalogue with cached, attributed source records.

    python3 scripts/build_book_catalog.py
    python3 scripts/build_book_catalog.py --offline  # reproduce from checked-in cache

No model calls, training data writes, account access, or paid API requests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.catalog.openlibrary import OpenLibraryClient, apply_review, digest, make_book

SUBJECTS = [
    "science_fiction",
    "fantasy",
    "mystery_and_detective_stories",
    "love_stories",
    "historical_fiction",
    "horror_tales",
    "biography",
    "science",
    "philosophy",
    "poetry",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--per-subject", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/catalog/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "web/data/books.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/catalog/manifest.json")
    parser.add_argument("--review", type=Path, default=ROOT / "data/catalog/selection_review.json")
    args = parser.parse_args()
    if not 1 <= args.per_subject <= 20:
        parser.error(
            "--per-subject must be 1..20; this importer is for bounded discovery, not bulk ingestion"
        )
    client = OpenLibraryClient(args.cache_dir, offline=args.offline)
    review = json.loads(args.review.read_text())
    books = {}
    queries = []
    rejected = []
    fields = "key,title,author_name,author_key,first_publish_year,cover_i,subject,language"
    for subject in SUBJECTS:
        response = client.get(
            "/search.json",
            {
                "q": f"subject:{subject} language:eng",
                "limit": args.per_subject,
                "fields": fields,
            },
        )
        queries.append({"url": response["url"], "sha256": response["sha256"]})
        for result in response["data"].get("docs", []):
            key = result.get("key", "")
            if key in books:
                continue
            try:
                work = client.get(key + ".json")
                book = apply_review(make_book(result, work), review)
                if book is None:
                    rejected.append(
                        {
                            "work": key,
                            "reason": review["excludedWorks"][key.split("/")[-1]]["reason"],
                        }
                    )
                else:
                    books[key] = book
            except ValueError as error:
                rejected.append({"work": key, "reason": str(error)})
        print(f"{subject}: {len(books)} verified work records", flush=True)
    catalogue = sorted(books.values(), key=lambda book: book["id"])
    if not catalogue:
        raise RuntimeError("No verified book records; refusing to replace catalogue")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "schemaVersion": 1,
        "source": "https://openlibrary.org/developers/api",
        "metadataLicense": "https://openlibrary.org/developers/licensing",
        "build": "python3 scripts/build_book_catalog.py --offline",
        "selection": "Bounded English-language subject search results; not a representative or exhaustive catalogue.",
        "count": len(catalogue),
        "withDescription": sum(bool(book["description"]) for book in catalogue),
        "withCover": sum(bool(book["coverUrl"]) for book in catalogue),
        "validatedMoods": 0,
        "catalogueSha256": digest(catalogue),
        "selectionReviewSha256": digest(review),
        "queries": queries,
        "rejected": rejected,
        "provenance": {
            "title_description_subjects": "Open Library work record (search subjects only if work subjects absent)",
            "authors_firstPublished": "Open Library search document for the same verified work and author IDs",
            "genres": "Deterministic exact-subject mapping in src/catalog/openlibrary.py",
            "moods": "Empty: no validated editorial mood source",
            "isbns": "Empty: edition-level ISBNs are intentionally not flattened into a work identifier",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Wrote {len(catalogue)} books ({manifest['withDescription']} source descriptions); {client.requests} network requests"
    )


if __name__ == "__main__":
    main()
