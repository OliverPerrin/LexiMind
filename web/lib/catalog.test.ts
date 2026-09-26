import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { books, readCatalogSnapshot, validateCatalog } from "./catalog";

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

test("malformed identifiers, dates, scalar fields and cover URLs fail closed", () => {
  for (const patch of [
    { identifiers: { openLibraryWork: 123, isbns: [] } },
    { source: { ...books[0].source, retrievedAt: 0 } },
    { firstPublished: true }, { coverUrl: false },
    { coverUrl: "https://covers.openlibrary.org.evil.example/b/id/1-L.jpg" },
    { authors: [" "] }, { genres: ["Fiction", "Fiction"] },
  ]) assert.throws(() => validateCatalog([{ ...books[0], ...patch }]));
});

test("publication receipts reject valid but mismatched catalogue snapshots", () => {
  const bytes = Buffer.from(JSON.stringify([books[0]]));
  const receipt = { schemaVersion: 1, count: 1, catalogueFileSha256: createHash("sha256").update(bytes).digest("hex") };
  assert.equal(readCatalogSnapshot(bytes, receipt).length, 1);
  assert.throws(() => readCatalogSnapshot(Buffer.from(JSON.stringify([books[1]])), receipt), /hash mismatch/);
  assert.throws(() => readCatalogSnapshot(bytes, { ...receipt, count: 2 }), /count mismatch/);
  assert.throws(() => readCatalogSnapshot(bytes, { ...receipt, schemaVersion: 2 }), /receipt/);
});

test("archive-only fields are not serialized into browser book props", () => {
  const result = validateCatalog([{ ...books[0], sourceContentHash: "abc", rawSource: { verbose: true } }]);
  assert.ok(!("sourceContentHash" in result[0]));
  assert.ok(!("rawSource" in result[0]));
});

test("source times require a valid calendar date and an explicit timezone", () => {
  for (const retrievedAt of ["2026-02-31T00:00:00Z", "2026-02-29T00:00:00Z", "01/02/2026", "2026-09-26T12:00:00", "2026-09-26T24:00:00Z"]) {
    assert.throws(() => validateCatalog([{ ...books[0], source: { ...books[0].source, retrievedAt } }]));
  }
  for (const retrievedAt of ["2024-02-29T00:00:00Z", "2026-09-26T12:00:00.123456+02:00"]) {
    assert.equal(validateCatalog([{ ...books[0], source: { ...books[0].source, retrievedAt } }]).length, 1);
  }
});
