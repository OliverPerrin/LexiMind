// This module belongs to the server graph. Only validated, public Book props cross into the browser.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Book } from "./types";

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function text(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}
function strings(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(text) && new Set(value).size === value.length;
}

function timestamp(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,6})?(Z|[+-]\d{2}:\d{2})$/.exec(value);
  if (!match) return false;
  const [year, month, day, hour, minute, second] = match.slice(1, 7).map(Number);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const monthDays = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return year >= 1 && month >= 1 && month <= 12 && day >= 1 && day <= monthDays[month - 1] &&
    hour <= 23 && minute <= 59 && second <= 59 && Number.isFinite(Date.parse(value));
}

/** Reject unsupported identities/provenance; expose only fields used by the website. */
export function validateCatalog(input: unknown): Book[] {
  if (!Array.isArray(input) || !input.length) throw new Error("The catalogue must contain books.");
  const ids = new Set<string>();
  return input.map((entry: unknown) => {
    if (!object(entry)) throw new Error("Invalid catalogue record: expected object");
    const source = entry.source;
    const identifiers = entry.identifiers;
    if (
      typeof entry.id !== "string" || !/^OL\d+W$/.test(entry.id) || ids.has(entry.id) ||
      !text(entry.title) || !strings(entry.authors) || !entry.authors.length ||
      !strings(entry.genres) || !strings(entry.subjects) || !strings(entry.moods) ||
      typeof entry.description !== "string" ||
      !object(source) || source.name !== "Open Library" ||
      source.url !== `https://openlibrary.org/works/${entry.id}` ||
      !timestamp(source.retrievedAt) ||
      !object(identifiers) || typeof identifiers.openLibraryWork !== "string" ||
      identifiers.openLibraryWork.replace(/^\/works\//, "") !== entry.id ||
      !strings(identifiers.isbns) ||
      !(entry.firstPublished === null || typeof entry.firstPublished === "number" && Number.isInteger(entry.firstPublished))
    ) throw new Error(`Invalid or duplicate catalogue identity: ${entry.id ?? "unknown"}`);

    if (entry.coverUrl !== null && (typeof entry.coverUrl !== "string" ||
      !/^https:\/\/covers\.openlibrary\.org\/b\/id\/\d+-[SML]\.jpg$/.test(entry.coverUrl))) {
      throw new Error(`Unrecognized cover source: ${entry.id}`);
    }
    if (entry.description && entry.descriptionSource !== source.url) {
      throw new Error(`Description lacks matching work provenance: ${entry.id}`);
    }
    if (entry.moods.length) {
      throw new Error(`Mood labels require a reviewed evidence schema before publication: ${entry.id}`);
    }
    ids.add(entry.id);
    // Unknown source fields/hashes stay in the archive, reducing the browser payload.
    return {
      id: entry.id, title: entry.title, authors: [...entry.authors],
      description: entry.description, descriptionSource: entry.description ? source.url as string : null,
      coverUrl: entry.coverUrl as string | null, genres: [...entry.genres],
      subjects: [...entry.subjects], moods: [], firstPublished: entry.firstPublished as number | null,
      source: { name: "Open Library", url: source.url as string, retrievedAt: source.retrievedAt },
      identifiers: { openLibraryWork: identifiers.openLibraryWork, isbns: [...identifiers.isbns] },
    };
  });
}

/** The receipt catches interrupted publication of the catalogue/manifest pair. */
export function readCatalogSnapshot(bytes: Uint8Array, receipt: unknown): Book[] {
  if (!object(receipt) || receipt.schemaVersion !== 1 || !Number.isInteger(receipt.count) ||
    typeof receipt.catalogueFileSha256 !== "string" || !/^[a-f0-9]{64}$/.test(receipt.catalogueFileSha256)) {
    throw new Error("Invalid catalogue publication receipt");
  }
  if (createHash("sha256").update(bytes).digest("hex") !== receipt.catalogueFileSha256) {
    throw new Error("Catalogue publication hash mismatch; rebuild both data files before deploying");
  }
  const books = validateCatalog(JSON.parse(new TextDecoder().decode(bytes)));
  if (books.length !== receipt.count) throw new Error("Catalogue publication count mismatch");
  return books;
}

export const books = readCatalogSnapshot(
  readFileSync(join(process.cwd(), "data/books.json")),
  JSON.parse(readFileSync(join(process.cwd(), "data/catalog-manifest.json"), "utf8")),
);
const byId = new Map(books.map(book => [book.id, book]));
export const getBook = (id: string): Book | undefined => byId.get(id);
