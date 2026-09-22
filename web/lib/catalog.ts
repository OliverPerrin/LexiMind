import rawBooks from "../data/books.json";
import type { Book } from "./types";

/** Fail the build on broken identities rather than silently publishing a bad join. */
export function validateCatalog(input: unknown): Book[] {
  if (!Array.isArray(input) || !input.length) throw new Error("The catalogue must contain books.");
  const ids = new Set<string>();
  return input.map((entry: unknown) => {
    const book = entry as Book;
    const strings = (value: unknown): value is string[] =>
      Array.isArray(value) && value.every((item) => typeof item === "string" && item.trim());
    if (
      !book || typeof book.id !== "string" || !/^OL\d+W$/.test(book.id) || ids.has(book.id) ||
      typeof book.title !== "string" || !book.title.trim() ||
      !strings(book.authors) || !book.authors.length ||
      !strings(book.genres) || !strings(book.subjects) || !strings(book.moods) ||
      typeof book.description !== "string" ||
      !book.source || book.source.url !== `https://openlibrary.org/works/${book.id}` ||
      !Number.isFinite(Date.parse(book.source.retrievedAt)) ||
      book.identifiers?.openLibraryWork.replace(/^\/works\//, "") !== book.id ||
      !strings(book.identifiers?.isbns) ||
      !(book.firstPublished === null || Number.isInteger(book.firstPublished))
    ) {
      throw new Error(`Invalid or duplicate catalogue identity: ${book?.id ?? "unknown"}`);
    }
    if (book.coverUrl && !/^https:\/\/covers\.openlibrary\.org\//.test(book.coverUrl)) {
      throw new Error(`Unrecognized cover source: ${book.id}`);
    }
    if (book.description && book.descriptionSource !== book.source.url) {
      throw new Error(`Description lacks matching work provenance: ${book.id}`);
    }
    if (book.moods.length) {
      throw new Error(`Mood labels require a reviewed evidence schema before publication: ${book.id}`);
    }
    ids.add(book.id);
    return book;
  });
}

export const books: Book[] = validateCatalog(rawBooks);
export const getBook = (id: string): Book | undefined => books.find((book) => book.id === id);
