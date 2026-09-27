import type { Book, Recommendation, RecommendationOptions, SearchResults } from "./types";

const STOP_WORDS = new Set(
  "a an the and or of in on at to for from with by about is are was were be been being i me my we you your it its they their this that these those book books read reading something want looking find would like please some more novel novels story stories where which who as into through than very also have has can could should".split(" "),
);

const normalize = (value: string) => value.normalize("NFKD")
  .replace(/[\u0300-\u036f]/g, "").toLowerCase()
  .replace(/\bsci[\s-]+fi\b/g, "science fiction");
const canonical = (word: string): string => {
  if (word.endsWith("ies") && word.length > 4) return `${word.slice(0, -3)}y`;
  if (word.endsWith("ing") && word.length > 6) return word.slice(0, -3);
  if (word.endsWith("ed") && word.length > 5) return word.slice(0, -2);
  if (word.endsWith("es") && word.length > 5 && !word.endsWith("ses")) return word.slice(0, -2);
  if (word.endsWith("s") && !word.endsWith("ss") && word.length > 4) return word.slice(0, -1);
  return word;
};

export function tokenize(text: string): string[] {
  return (normalize(text).match(/[\p{L}\p{N}]+/gu) ?? [])
    .filter(term => term.length > 1 && !STOP_WORDS.has(term)).map(canonical);
}

// Vocabulary aliases only; never inferred judgments about a book's mood.
const ALIASES: Record<string, string[]> = {
  scifi: ["science", "fiction"], nonfiction: ["nonfiction"],
  detective: ["mystery", "detective"], whodunit: ["mystery", "detective"],
  funny: ["humor", "humorous", "comedy"], humour: ["humor", "humorous", "comedy"],
  humorous: ["humor", "humorous", "comedy"], memoir: ["memoir", "autobiography"],
  space: ["space", "science"],
};

type Vector = Map<string, number>;
interface Metadata {
  title: string;
  titleTerms: Set<string>;
  authors: string[];
  genres: Set<string>;
  genreTerms: Set<string>[];
  subjects: Set<string>;
  moods: Set<string>;
}
interface Vectors { vectors: Vector[]; idf: Map<string, number> }
interface Candidate extends Recommendation { position: number; selected: boolean }
interface Ranking {
  candidates: Candidate[];
  selected: Recommendation[];
  authors: Map<string, number>;
  genres: Map<string, number>;
  diversifyGenres: boolean;
}
interface Index {
  byId: Map<string, number>;
  metadata: Metadata[];
  exclusions: { genre: string; pattern: RegExp }[];
  text?: Vectors;
  queries: Map<string, Ranking>;
}

// Catalogues are immutable snapshots. Weak keys release indexes with a released snapshot.
const indexes = new WeakMap<Book[], Index>();
const normalizedSet = (values: string[]) => new Set(values.map(normalize));
const unique = (values: string[] = []) => [...new Set(values)];
const canonicalFacet = (values: string[] = []) => [...normalizedSet(values)].sort();
const intersects = (left: Set<string>, right: Set<string>) => {
  for (const value of right) if (left.has(value)) return true;
  return false;
};

function getIndex(books: Book[]): Index {
  const cached = indexes.get(books);
  if (cached) return cached;
  const allGenres = [...new Set(books.flatMap(book => book.genres).map(normalize))]
    .sort((a, b) => b.length - a.length || a.localeCompare(b));
  const index: Index = {
    byId: new Map(books.map((book, i) => [book.id, i])),
    metadata: books.map(book => ({
      title: normalize(book.title), titleTerms: new Set(tokenize(book.title)),
      authors: book.authors.map(normalize), genres: normalizedSet(book.genres),
      genreTerms: book.genres.map(genre => new Set(tokenize(genre))),
      subjects: normalizedSet(book.subjects), moods: normalizedSet(book.moods),
    })),
    // Longest first: "without science fiction" must not become "without science" + "fiction".
    exclusions: allGenres.map(genre => ({
      genre,
      pattern: new RegExp(`\\b(?:without|no|not|exclude|excluding)\\s+(?:any\\s+)?${genre.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "g"),
    })),
    queries: new Map(),
  };
  indexes.set(books, index);
  return index;
}

function unit(vector: Vector): Vector {
  let squared = 0;
  for (const value of vector.values()) squared += value * value;
  const norm = Math.sqrt(squared);
  if (norm) for (const [term, value] of vector) vector.set(term, value / norm);
  return vector;
}

function textIndex(books: Book[], index: Index): Vectors {
  if (index.text) return index.text;
  const frequency = new Map<string, number>();
  const vectors = books.map(book => {
    const vector: Vector = new Map();
    for (const [text, weight] of [
      [book.title, 4], [book.authors.join(" "), 2], [book.genres.join(" "), 4],
      [book.subjects.join(" "), 2], [book.moods.join(" "), 2], [book.description, 1],
    ] as [string, number][]) {
      for (const term of tokenize(text)) vector.set(term, (vector.get(term) ?? 0) + weight);
    }
    for (const term of vector.keys()) frequency.set(term, (frequency.get(term) ?? 0) + 1);
    return vector;
  });
  const idf = new Map([...frequency].map(([term, count]) => [term, Math.log(1 + (books.length + 1) / (count + 1))]));
  for (const vector of vectors) {
    for (const [term, value] of vector) vector.set(term, (1 + Math.log(value)) * (idf.get(term) ?? 1));
    unit(vector);
  }
  index.text = { vectors, idf };
  return index.text;
}

function cosine(a: Vector, b: Vector): number {
  let result = 0;
  const [small, large] = a.size < b.size ? [a, b] : [b, a];
  for (const [term, value] of small) result += value * (large.get(term) ?? 0);
  return result;
}

function queryVector(terms: string[], idf: Map<string, number>): Vector {
  const vector: Vector = new Map();
  for (const token of terms) for (const term of Object.hasOwn(ALIASES, token) ? ALIASES[token] : [token]) {
    const word = canonical(term);
    const weight = idf.get(word);
    if (weight !== undefined) vector.set(word, (vector.get(word) ?? 0) + weight);
  }
  return unit(vector);
}

function scoreCandidates(books: Book[], index: Index, options: RecommendationOptions): Ranking {
  let query = normalize(options.query?.trim() ?? "");
  const excludedGenres = new Set<string>();
  for (const { genre, pattern } of index.exclusions) {
    pattern.lastIndex = 0;
    if (pattern.test(query)) { excludedGenres.add(genre); query = query.replace(pattern, " "); }
  }
  const terms = tokenize(query);
  const hasQuery = terms.length > 0;
  const normalizedQuery = query.trim();
  const toIndices = (ids: string[] = []) => unique(ids).flatMap(id => index.byId.has(id) ? [index.byId.get(id)!] : []);
  const seeds = toIndices(options.seedIds);
  const saved = toIndices(options.savedIds);
  const savedIds = new Set(options.savedIds ?? []);
  const hidden = new Set([...(options.dismissedIds ?? []), ...(options.seedIds ?? [])]);
  const wantedGenres = normalizedSet(options.genres ?? []);
  const wantedSubjects = normalizedSet(options.subjects ?? []);
  const wantedMoods = normalizedSet(options.moods ?? []);
  // Browsing/filtering needs metadata only; defer TF-IDF until content matching is requested.
  const text = hasQuery || seeds.length || saved.length ? textIndex(books, index) : undefined;
  const queryTerms = hasQuery ? queryVector(terms, text!.idf) : new Map<string, number>();
  const queryKeys = new Set(queryTerms.keys());
  const candidates: Candidate[] = [];

  books.forEach((book, position) => {
    const metadata = index.metadata[position];
    if (hidden.has(book.id) || intersects(metadata.genres, excludedGenres)) return;
    if (wantedGenres.size && !intersects(metadata.genres, wantedGenres)) return;
    if (wantedSubjects.size && !intersects(metadata.subjects, wantedSubjects)) return;
    if (wantedMoods.size && !intersects(metadata.moods, wantedMoods)) return;
    const vector = text?.vectors[position];
    const relevance = hasQuery ? cosine(queryTerms, vector!) : 0;
    if (hasQuery && relevance <= 0) return;
    let score = relevance * 3;
    const reasons: string[] = [];
    if (normalizedQuery.length > 1 && metadata.title === normalizedQuery) score += 3;
    else if (normalizedQuery.length > 2 && metadata.title.includes(normalizedQuery)) score += 1.25;
    const matchedGenres = book.genres.filter((_, i) => intersects(metadata.genreTerms[i], queryKeys));
    if (matchedGenres.length) reasons.push(matchedGenres.slice(0, 2).join(" · "));
    if (hasQuery && !reasons.length) {
      if (metadata.authors.some(author => author.includes(normalizedQuery))) reasons.push("Matches the author you searched for");
      else if (intersects(metadata.titleTerms, queryKeys)) reasons.push("Matches your title search");
      else reasons.push("Related subjects and description");
    }
    if (seeds.length) {
      let closest = seeds[0];
      let similarity = -Infinity;
      for (const seed of seeds) {
        const candidate = cosine(vector!, text!.vectors[seed]);
        if (candidate > similarity) { similarity = candidate; closest = seed; }
      }
      const sharedGenres = book.genres.filter(genre => index.metadata[closest].genres.has(normalize(genre)));
      const sharedSubject = book.subjects.find(subject => index.metadata[closest].subjects.has(normalize(subject)));
      score += similarity * 2 + Math.min(sharedGenres.length, 2) * 0.12;
      if (sharedGenres.length) reasons.push(`${sharedGenres[0]}, like ${books[closest].title}`);
      else if (sharedSubject) reasons.push(`Shared subject: ${sharedSubject}`);
      else if (similarity > 0.025) reasons.push(`Related description to ${books[closest].title}`);
      if (!hasQuery && !sharedGenres.length && similarity <= 0.025) return;
    }
    if (saved.length && !savedIds.has(book.id)) {
      let affinity = 0;
      for (const seed of saved) affinity = Math.max(affinity, cosine(vector!, text!.vectors[seed]));
      score += affinity * 0.4;
      if (!reasons.length && affinity > 0.08) reasons.push("Related to books on your shelf");
    }
    if (!hasQuery && !seeds.length) score += Math.min(book.description.length / 800, 1) * 0.05;
    if (!reasons.length && wantedSubjects.size) reasons.push(`Subject: ${book.subjects.find(subject => wantedSubjects.has(normalize(subject)))}`);
    if (!reasons.length && wantedGenres.size) reasons.push(book.genres.filter(genre => wantedGenres.has(normalize(genre))).join(" · "));
    if (!reasons.length) reasons.push(book.genres.slice(0, 2).join(" · ") || "From the catalogue");
    candidates.push({ book, score, reasons: reasons.slice(0, 2), position, selected: false });
  });

  candidates.sort((a, b) => b.score - a.score || a.position - b.position);
  return { candidates, selected: [], authors: new Map(), genres: new Map(), diversifyGenres: !hasQuery && !seeds.length && !wantedGenres.size };
}

function extendRanking(ranking: Ranking, limit: number): void {
  while (ranking.selected.length < limit) {
    let best: Candidate | undefined;
    let bestAdjusted = -Infinity;
    for (const candidate of ranking.candidates) {
      if (candidate.selected) continue;
      // Penalties are nonnegative; lower base scores cannot beat the best adjusted score.
      if (candidate.score < bestAdjusted) break;
      let authorRepeats = 0;
      for (const author of candidate.book.authors) authorRepeats = Math.max(authorRepeats, ranking.authors.get(author) ?? 0);
      const genreRepeats = ranking.genres.get(candidate.book.genres[0] ?? "") ?? 0;
      const penalty = authorRepeats * 0.035 + (ranking.diversifyGenres ? genreRepeats * 0.02 : 0);
      const adjusted = candidate.score - penalty;
      if (adjusted > bestAdjusted) { best = candidate; bestAdjusted = adjusted; }
    }
    if (!best) break;
    best.selected = true;
    ranking.selected.push({ book: best.book, score: best.score, reasons: best.reasons });
    for (const author of best.book.authors) ranking.authors.set(author, (ranking.authors.get(author) ?? 0) + 1);
    const genre = best.book.genres[0] ?? "";
    ranking.genres.set(genre, (ranking.genres.get(genre) ?? 0) + 1);
  }
}

/** Source-based ranking with an exact total and an incrementally ranked visible prefix. */
export function searchBooks(books: Book[], options: RecommendationOptions = {}): SearchResults {
  const index = getIndex(books);
  const key = JSON.stringify([
    normalize(options.query?.trim() ?? ""), canonicalFacet(options.genres),
    canonicalFacet(options.subjects), canonicalFacet(options.moods),
    unique(options.seedIds), unique(options.savedIds).sort(), unique(options.dismissedIds).sort(),
  ]);
  let ranking = index.queries.get(key);
  if (!ranking) {
    ranking = scoreCandidates(books, index, options);
    // Bound retained query/personalization state. No persistent storage or network use.
    if (index.queries.size >= 8) index.queries.delete(index.queries.keys().next().value!);
  } else index.queries.delete(key);
  index.queries.set(key, ranking);
  const limit = Number.isFinite(options.limit)
    ? Math.min(ranking.candidates.length, Math.max(0, Math.floor(options.limit!)))
    : ranking.candidates.length;
  extendRanking(ranking, limit);
  return {
    // Copy result containers so callers cannot mutate cached ordering/reasons.
    items: ranking.selected.slice(0, limit).map(item => ({ ...item, reasons: [...item.reasons] })),
    total: ranking.candidates.length,
  };
}

/** Compatibility helper for existing callers that need only the ranked items. */
export function recommendBooks(books: Book[], options: RecommendationOptions = {}): Recommendation[] {
  return searchBooks(books, options).items;
}
