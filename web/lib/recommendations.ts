import type { Book, Recommendation, RecommendationOptions } from "./types";

const STOP_WORDS = new Set(
  "a an the and or of in on at to for from with by about is are was were be been being i me my we you your it its they their this that these those book books read reading something want looking find would like please some more novel novels story stories where which who as into through than very also have has can could should".split(" "),
);

const normalize = (value: string) => value.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase()
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
    .filter((term) => term.length > 1 && !STOP_WORDS.has(term))
    .map(canonical);
}

// These aliases normalize vocabulary, not emotional judgments about a book.
const ALIASES: Record<string, string[]> = {
  scifi: ["science", "fiction"],
  nonfiction: ["nonfiction"],
  detective: ["mystery", "detective"],
  whodunit: ["mystery", "detective"],
  funny: ["humor", "humorous", "comedy"],
  humour: ["humor", "humorous", "comedy"],
  humorous: ["humor", "humorous", "comedy"],
  memoir: ["memoir", "autobiography"],
  space: ["space", "science"],
};

type Vector = Map<string, number>;
interface Index {
  vectors: Vector[];
  idf: Map<string, number>;
  byId: Map<string, number>;
}
const indexCache = new WeakMap<Book[], Index>();

function counts(text: string, weight: number, into: Vector): void {
  for (const term of tokenize(text)) into.set(term, (into.get(term) ?? 0) + weight);
}

function unit(vector: Vector): Vector {
  const norm = Math.sqrt([...vector.values()].reduce((sum, value) => sum + value * value, 0));
  if (norm) for (const [term, value] of vector) vector.set(term, value / norm);
  return vector;
}

function cosine(a: Vector, b: Vector): number {
  let result = 0;
  const [small, large] = a.size < b.size ? [a, b] : [b, a];
  for (const [term, value] of small) result += value * (large.get(term) ?? 0);
  return result;
}

function buildIndex(books: Book[]): Index {
  const cached = indexCache.get(books);
  if (cached) return cached;
  const frequency = new Map<string, number>();
  const vectors = books.map((book) => {
    const vector: Vector = new Map();
    counts(book.title, 4, vector);
    counts(book.authors.join(" "), 2, vector);
    counts(book.genres.join(" "), 4, vector);
    counts(book.subjects.join(" "), 2, vector);
    counts(book.moods.join(" "), 2, vector);
    counts(book.description, 1, vector);
    for (const term of vector.keys()) frequency.set(term, (frequency.get(term) ?? 0) + 1);
    return vector;
  });
  const idf = new Map([...frequency].map(([term, count]) => [term, Math.log(1 + (books.length + 1) / (count + 1))]));
  for (const vector of vectors) {
    for (const [term, value] of vector) vector.set(term, (1 + Math.log(value)) * (idf.get(term) ?? 1));
    unit(vector);
  }
  const index = { vectors, idf, byId: new Map(books.map((book, i) => [book.id, i])) };
  indexCache.set(books, index);
  return index;
}

function queryVector(query: string, index: Index): Vector {
  const vector: Vector = new Map();
  for (const token of tokenize(query)) {
    for (const term of ALIASES[token] ?? [token]) {
      const word = canonical(term);
      if (index.idf.has(word)) vector.set(word, (vector.get(word) ?? 0) + (index.idf.get(word) ?? 1));
    }
  }
  return unit(vector);
}

function overlap(a: string[], b: string[]): string[] {
  const right = new Set(b.map(normalize));
  return a.filter((value) => right.has(normalize(value)));
}

function exclusions(query: string, books: Book[]): { query: string; genres: Set<string> } {
  const genres = new Set<string>();
  let positive = normalize(query);
  for (const genre of new Set(books.flatMap((book) => book.genres))) {
    const escaped = normalize(genre).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = new RegExp(`\\b(?:without|no|not|exclude|excluding)\\s+(?:any\\s+)?${escaped}\\b`, "g");
    if (pattern.test(positive)) {
      genres.add(normalize(genre));
      positive = positive.replace(pattern, " ");
    }
  }
  return { query: positive, genres };
}

/** Deterministic content-based baseline. No ratings or research predictions are fabricated. */
export function recommendBooks(books: Book[], options: RecommendationOptions = {}): Recommendation[] {
  const index = buildIndex(books);
  const { query, genres: excludedGenres } = exclusions(options.query?.trim() ?? "", books);
  const terms = tokenize(query);
  const queryTerms = queryVector(query, index);
  const hasQuery = terms.length > 0;
  const seeds = (options.seedIds ?? []).flatMap((id) => index.byId.has(id) ? [index.byId.get(id)!] : []);
  const saved = (options.savedIds ?? []).flatMap((id) => index.byId.has(id) ? [index.byId.get(id)!] : []);
  const hidden = new Set([...(options.dismissedIds ?? []), ...(options.seedIds ?? [])]);
  const wantedGenres = options.genres ?? [];
  const wantedMoods = options.moods ?? [];

  const candidates: Recommendation[] = [];
  books.forEach((book, i) => {
    if (hidden.has(book.id) || book.genres.some((genre) => excludedGenres.has(normalize(genre)))) return;
    if (wantedGenres.length && !overlap(book.genres, wantedGenres).length) return;
    if (wantedMoods.length && !overlap(book.moods, wantedMoods).length) return;
    const vector = index.vectors[i];
    const relevance = hasQuery ? cosine(queryTerms, vector) : 0;
    if (hasQuery && relevance <= 0) return;
    let score = relevance * 3;
    const reasons: string[] = [];
    const normalizedTitle = normalize(book.title);
    const normalizedQuery = normalize(query.trim());
    if (normalizedQuery.length > 1 && normalizedTitle === normalizedQuery) score += 3;
    else if (normalizedQuery.length > 2 && normalizedTitle.includes(normalizedQuery)) score += 1.25;
    const matchedGenres = book.genres.filter((genre) => tokenize(genre).some((term) => queryTerms.has(term)));
    if (matchedGenres.length) reasons.push(matchedGenres.slice(0, 2).join(" · "));
    if (hasQuery && !reasons.length) {
      if (book.authors.some((author) => normalize(author).includes(normalizedQuery))) reasons.push("Matches the author you searched for");
      else if (tokenize(book.title).some((term) => queryTerms.has(term))) reasons.push("Matches your title search");
      else reasons.push("Related subjects and description");
    }
    if (seeds.length) {
      const closest = seeds.map((seed) => ({ seed, similarity: cosine(vector, index.vectors[seed]) }))
        .sort((a, b) => b.similarity - a.similarity)[0];
      const sharedGenres = overlap(book.genres, books[closest.seed].genres);
      const sharedSubjects = overlap(book.subjects, books[closest.seed].subjects);
      score += closest.similarity * 2 + Math.min(sharedGenres.length, 2) * 0.12;
      if (sharedGenres.length) reasons.push(`${sharedGenres[0]}, like ${books[closest.seed].title}`);
      else if (sharedSubjects.length) reasons.push(`Shared subject: ${sharedSubjects[0]}`);
      else if (closest.similarity > 0.025) reasons.push(`Related description to ${books[closest.seed].title}`);
      if (!hasQuery && !sharedGenres.length && closest.similarity <= 0.025) return;
    }
    if (saved.length && !options.savedIds?.includes(book.id)) {
      const affinity = Math.max(...saved.map((seed) => cosine(vector, index.vectors[seed])));
      score += affinity * 0.4;
      if (!reasons.length && affinity > 0.08) reasons.push("Related to books on your shelf");
    }
    if (!hasQuery && !seeds.length) {
      // Prefer informative records; source order is the stable tiebreaker, not an invented popularity score.
      score += Math.min(book.description.length / 800, 1) * 0.05;
    }
    if (!reasons.length && wantedGenres.length) reasons.push(overlap(book.genres, wantedGenres).join(" · "));
    if (!reasons.length) reasons.push(book.genres.slice(0, 2).join(" · ") || "From the catalogue");
    candidates.push({ book, score, reasons: reasons.slice(0, 2) });
  });

  candidates.sort((a, b) => b.score - a.score || (index.byId.get(a.book.id)! - index.byId.get(b.book.id)!));
  const limit = Number.isFinite(options.limit) ? Math.max(0, Math.floor(options.limit!)) : candidates.length;
  const selected: Recommendation[] = [];
  const authorCounts = new Map<string, number>();
  const genreCounts = new Map<string, number>();
  while (candidates.length && selected.length < limit) {
    let best = 0;
    let bestAdjusted = -Infinity;
    candidates.forEach((candidate, i) => {
      const authorRepeats = Math.max(0, ...candidate.book.authors.map((author) => authorCounts.get(author) ?? 0));
      const genreRepeats = genreCounts.get(candidate.book.genres[0] ?? "") ?? 0;
      const penalty = authorRepeats * 0.035 + ((!hasQuery && !seeds.length && !wantedGenres.length) ? genreRepeats * 0.02 : 0);
      const adjusted = candidate.score - penalty;
      if (adjusted > bestAdjusted) { best = i; bestAdjusted = adjusted; }
    });
    const [choice] = candidates.splice(best, 1);
    selected.push(choice);
    for (const author of choice.book.authors) authorCounts.set(author, (authorCounts.get(author) ?? 0) + 1);
    const genre = choice.book.genres[0] ?? "";
    genreCounts.set(genre, (genreCounts.get(genre) ?? 0) + 1);
  }
  return selected;
}
