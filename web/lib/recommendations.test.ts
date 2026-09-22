import assert from "node:assert/strict";
import test from "node:test";
import { recommendBooks } from "./recommendations";
import type { Book } from "./types";

function book(id: number, title: string, genre: string, description: string, author = "Author"): Book {
  return {
    id: `OL${id}W`, title, authors: [author], description, genres: [genre],
    subjects: [], moods: [], coverUrl: null, firstPublished: 2000,
    source: { name: "Open Library", url: `https://openlibrary.org/works/OL${id}W`, retrievedAt: "2026-09-22T00:00:00Z" },
    identifiers: { openLibraryWork: `OL${id}W`, isbns: [] },
  };
}
const catalogue = [
  book(1, "Voyage Among the Stars", "Science fiction", "An astronaut explores distant planets and befriends an alien scientist.", "Mira Chen"),
  book(2, "The Planetary Garden", "Science fiction", "Scientists grow a garden on a distant planet while exploring the stars.", "Sana Patel"),
  book(3, "The Missing Letter", "Mystery", "A detective investigates a murder in a secluded country house.", "Morgan Bell"),
  book(4, "Letters to June", "Romance", "A musician and a painter fall in love in a seaside town.", "Alex Stone"),
  book(5, "Murder at the Observatory", "Mystery", "A detective investigates a murder among scientists studying planets.", "Robin Dale"),
];

test("exact title wins over broad description overlap", () => {
  const results = recommendBooks(catalogue, { query: "The Missing Letter" });
  assert.equal(results[0].book.id, "OL3W");
  assert.ok(results[0].reasons.length > 0);
});

test("natural phrasing drops filler and matches book content", () => {
  const results = recommendBooks(catalogue, { query: "I want to read about an astronaut exploring planets" });
  assert.equal(results[0].book.id, "OL1W");
  assert.ok(!results.some((result) => result.book.id === "OL4W"));
});

test("author searches work with accents and case normalization", () => {
  assert.equal(recommendBooks(catalogue, { query: "MÍRA CHEN" })[0].book.id, "OL1W");
});

test("no matches returns an honest empty result", () => {
  assert.deepEqual(recommendBooks(catalogue, { query: "xyzzynonexistent" }), []);
});

test("facets and dismissals are hard constraints, not score penalties", () => {
  const results = recommendBooks(catalogue, { genres: ["Mystery"], dismissedIds: ["OL3W"] });
  assert.deepEqual(results.map((result) => result.book.id), ["OL5W"]);
  assert.deepEqual(recommendBooks(catalogue, { genres: ["History"] }), []);
});

test("explicit genre exclusions do not become positive query signals", () => {
  const results = recommendBooks(catalogue, { query: "something to read without romance" });
  assert.equal(results.length, 4);
  assert.ok(results.every((result) => !result.book.genres.includes("Romance")));
});

test("compound science-fiction vocabulary works in queries and exclusions", () => {
  const results = recommendBooks(catalogue, { query: "sci-fi" });
  assert.ok(results.slice(0, 2).every((result) => result.book.genres.includes("Science fiction")));
  const excluded = recommendBooks(catalogue, { query: "without sci-fi" });
  assert.ok(excluded.length > 0);
  assert.ok(excluded.every((result) => !result.book.genres.includes("Science fiction")));
});

test("more like this excludes its seed and prefers related content", () => {
  const results = recommendBooks(catalogue, { seedIds: ["OL1W"] });
  assert.equal(results[0].book.id, "OL2W");
  assert.ok(results.every((result) => result.book.id !== "OL1W"));
  assert.ok(results[0].reasons.some((reason) => reason.includes("Voyage Among the Stars")));
});

test("saved books influence recommendations without fabricating ratings", () => {
  const results = recommendBooks(catalogue, { savedIds: ["OL3W"], dismissedIds: ["OL3W"] });
  assert.equal(results[0].book.id, "OL5W");
  assert.ok(results.every((result) => Number.isFinite(result.score)));
});

test("unvalidated moods are never inferred by the ranker", () => {
  assert.deepEqual(recommendBooks(catalogue, { moods: ["Hopeful"] }), []);
});

test("results are deterministic, bounded, and leave source books unchanged", () => {
  const before = JSON.stringify(catalogue);
  assert.deepEqual(recommendBooks(catalogue, { limit: 3 }), recommendBooks(catalogue, { limit: 3 }));
  assert.equal(recommendBooks(catalogue, { limit: 3 }).length, 3);
  assert.deepEqual(recommendBooks(catalogue, { limit: 0 }), []);
  assert.equal(JSON.stringify(catalogue), before);
});
