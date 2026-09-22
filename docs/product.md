# LexiMind: book discovery

Status: implementation underway, 22 September 2026. Research training and experiments
are paused at the owner's request. Software tests and catalogue verification remain
part of development.

## Purpose

Help a reader find their next book through subjects, genres, books they already like,
and eventually well-supported descriptions of mood. The website and the research
programme have separate release criteria: a useful catalogue can ship before a new
model has been trained.

## First release

- A small catalogue with stable Open Library work IDs, authors, source descriptions,
  subjects, mapped genres, covers, publication years, and links to the source records.
- Search by title, author, or a description of what to read.
- Content-based recommendations using text similarity and explicit metadata, with
  grounded reasons and modest author/genre diversity.
- More like this, favourites, a reading list, hidden books, and a way to restore them.
- Preferences stay in browser local storage. No account or cross-device sync.
- A responsive Next.js application deployable independently from Python research.

The current search baseline uses weighted TF-IDF and cosine similarity, with a small
vocabulary normalization layer. It is not a neural semantic search model. Scores are
relative ranking values, never ratings, confidence, or a measure of book quality.
The site makes no recommendation-quality claim from unit tests.

## Data contract

`web/data/books.json` is the deployment snapshot. Its importer lives in
`src/catalog/` and `scripts/build_book_catalog.py`; the source manifest and review
decisions live in `data/catalog/`. Raw API cache records include retrieval time,
source URL, response hash, and full source data. Community metadata can still contain
errors: source attribution makes correction possible; identity checks do not prove
every cataloguer's statement true.

An Open Library work groups editions. Cover and original publication year may come
from different editions of that work and must not be presented as one verified
edition. ISBNs are left empty when no particular edition was selected.

Description text stays attached to its identified work. No title-only joins with
Gutenberg or Goodreads are allowed. Unverifiable legacy pairs remain historical
artifacts and are excluded from the new catalogue. There are no generated book
descriptions in this release.

Genres map explicit source subjects to display labels. Mood labels are empty until
independently supported; the website hides that filter when no eligible labels exist.
The legacy emotion head was trained on Reddit text and does not establish a book's
mood. Randomly sampled tones must never enter the catalogue or ranking features.

## Deployment and privacy

The web app ships a fixed catalogue snapshot. It does not load model checkpoints,
contact a paid model API, collect browsing events, or require database credentials.
Book covers load from Open Library, and source links navigate there. The About page
explains this and the local nature of saved preferences.

Deploy only the `web/` directory to Vercel. This keeps checkpoint, experiment, raw
training-data, and local credential files out of the deployment. `npm ci` and the
committed package lock establish the web dependencies.

## Resume points

1. Add more independently identified works while reviewing rejected source records.
2. Add optional export/import of a reading list, then accounts only if cross-device
   use becomes important.
3. Define a book-domain mood annotation guide with evidence and uncertainty before
   enabling mood filtering. A genre, a character's expressed emotion, and a reader's
   experience are different targets.
4. Prepare a held-out-by-work set of queries and relevance judgments. Human relevance
   judgments, not implementation tests, determine recommendation quality.
5. Once research is resumed, compare a pretrained embedding retrieval baseline with
   MTL-derived features. Keep judgement/evaluation data out of training and model
   selection. See `docs/eval_protocol.md` for the draft research protocol.

There is no paid compute or training requirement for browsing this release.
