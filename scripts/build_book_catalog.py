"""Build the books-first website catalogue with cached, attributed source records.

    python3 scripts/build_book_catalog.py
    python3 scripts/build_book_catalog.py --offline  # reproduce from checked-in cache

No model calls, training data writes, account access, or paid API requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.catalog.openlibrary import (
    CatalogueIntegrityError,
    OpenLibraryClient,
    SourceRecordError,
    apply_review,
    digest,
    make_book,
    validate_catalogue,
    validate_review,
)
from src.catalog.storage import json_text, publish_catalogue

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


EXTRA_QUERIES = [
    ("popular_science", 'subject:"popular science" language:eng'),
    ("literary_fiction", 'subject:"literary fiction" language:eng'),
    ("essays", "subject:essays language:eng"),
]


def build_catalogue(
    client: OpenLibraryClient, review: dict, *, per_subject: int = 10
) -> tuple[list[dict], dict]:
    if not 1 <= per_subject <= 20:
        raise ValueError("per_subject must be 1..20")
    validate_review(review)
    books = {}
    queries = []
    rejected = []
    seen = set()
    fields = "key,title,author_name,author_key,first_publish_year,cover_i,subject,language"
    selections = [(subject, f"subject:{subject} language:eng", per_subject) for subject in SUBJECTS]
    selections += [(label, query, 6) for label, query in EXTRA_QUERIES]
    for label, query, limit in selections:
        response = client.get("/search.json", {"q": query, "limit": limit, "fields": fields})
        results = response["data"].get("docs")
        if not isinstance(results, list):
            raise CatalogueIntegrityError(f"Search response has no result array: {response['url']}")
        if len(results) > limit:
            raise CatalogueIntegrityError("Search response exceeded the bounded selection limit")
        queries.append({"url": response["url"], "sha256": response["sha256"]})
        for result in results:
            if not isinstance(result, dict) or not isinstance(result.get("key"), str):
                rejected.append({"work": None, "reason": "Missing work identifier"})
                continue
            key = result["key"]
            if key in seen:
                continue
            seen.add(key)
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
            except SourceRecordError as error:
                # Missing ordinary metadata is a record rejection. Cache, hash,
                # review, and source-shape failures abort the entire build.
                rejected.append({"work": key, "reason": str(error)})
        print(f"{label}: {len(books)} verified work records", flush=True)
    catalogue = sorted(books.values(), key=lambda book: book["id"])
    validate_catalogue(catalogue)
    manifest = {
        "schemaVersion": 1,
        "source": "https://openlibrary.org/developers/api",
        "metadataLicense": "https://openlibrary.org/developers/licensing",
        "build": f"python3 scripts/build_book_catalog.py --offline --per-subject {per_subject}",
        "buildOptions": {"perSubject": per_subject, "extraQueryLimit": 6},
        "selection": "Bounded English-language subject search results; not a representative or exhaustive catalogue.",
        "count": len(catalogue),
        "withDescription": sum(bool(book["description"]) for book in catalogue),
        "withCover": sum(bool(book["coverUrl"]) for book in catalogue),
        "validatedMoods": 0,
        "catalogueSha256": digest(catalogue),
        "catalogueFileSha256": hashlib.sha256(json_text(catalogue).encode()).hexdigest(),
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
    validate_catalogue(catalogue, manifest)
    return catalogue, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--per-subject", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/catalog/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "web/data/books.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--review", type=Path, default=ROOT / "data/catalog/selection_review.json")
    args = parser.parse_args()
    if not 1 <= args.per_subject <= 20:
        parser.error(
            "--per-subject must be 1..20; this importer is for bounded discovery, not bulk ingestion"
        )
    client = OpenLibraryClient(args.cache_dir, offline=args.offline)
    review = json.loads(args.review.read_text(encoding="utf-8"))
    catalogue, manifest = build_catalogue(client, review, per_subject=args.per_subject)
    default_output = ROOT / "web/data/books.json"
    manifest_path = args.manifest or (
        ROOT / "data/catalog/manifest.json"
        if args.output.resolve() == default_output.resolve()
        else args.output.with_name("manifest.json")
    )
    receipt_path = args.receipt or args.output.with_name("catalog-manifest.json")
    publish_catalogue(args.output, catalogue, manifest_path, manifest, receipt_path)
    print(
        f"Wrote {len(catalogue)} books ({manifest['withDescription']} source descriptions); {client.requests} network requests"
    )


if __name__ == "__main__":
    main()
