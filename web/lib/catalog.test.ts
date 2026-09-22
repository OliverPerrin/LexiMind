import assert from "node:assert/strict";
import test from "node:test";
import { books, validateCatalog } from "./catalog";

test("the deployment snapshot has unique identified works and attributed descriptions", () => {
  assert.equal(new Set(books.map((book) => book.id)).size, books.length);
  assert.ok(books.every((book) => !book.description || book.descriptionSource === book.source.url));
});

test("a build cannot publish a description attached to the wrong work", () => {
  assert.throws(() => validateCatalog([{ ...books[0], descriptionSource: "https://openlibrary.org/works/OL999999W" }]), /provenance/);
});

test("a build rejects invented moods until an evidence schema is reviewed", () => {
  assert.throws(() => validateCatalog([{ ...books[0], moods: ["Hopeful"] }]), /evidence schema/);
});

test("duplicate works and unknown source links cannot be deployed", () => {
  assert.throws(() => validateCatalog([books[0], books[0]]), /duplicate/);
  assert.throws(() => validateCatalog([{ ...books[0], source: { ...books[0].source, url: "javascript:alert(1)" } }]), /identity/);
});
