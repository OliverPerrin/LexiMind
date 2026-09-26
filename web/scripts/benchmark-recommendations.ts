/** Engineering microbenchmark only: synthetic metadata, no models or relevance judgments. */
import { performance } from "node:perf_hooks";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import type { Book, RecommendationOptions } from "../lib/types";

function catalogue(size: number): Book[] {
  const genres = ["Science fiction", "Mystery", "History", "Romance", "Fantasy"];
  return Array.from({ length: size }, (_, i) => ({
    id: `OL${i + 1}W`, title: `Synthetic book ${i + 1}`,
    authors: [`Synthetic author ${i % 173}`],
    description: `A story of ${i % 2 ? "exploration planets science" : "friendship family history"}. ` +
      "Its characters discover distant places, investigate mysteries, and learn about their world. ".repeat(4),
    genres: [genres[i % genres.length]], subjects: [`Subject ${i % 23}`, "Literature"],
    moods: [], coverUrl: null, firstPublished: 2000,
    source: { name: "Synthetic fixture", url: `https://example.org/${i}`, retrievedAt: "2026-09-26T00:00:00Z" },
    identifiers: { openLibraryWork: `/works/OL${i + 1}W`, isbns: [] },
  }));
}

async function main() {
  const modulePath = resolve(process.argv[2] ?? "lib/recommendations.ts");
  const engine = await import(pathToFileURL(modulePath).href);
  const newApi = typeof engine.searchBooks === "function";
  const report: unknown[] = [];
  for (const size of [89, 1000, 5000]) {
    for (const [name, options] of [
      ["browse", {}], ["search", { query: "exploration planets" }],
      ["similar", { seedIds: ["OL1W"] }],
    ] as [string, RecommendationOptions][]) {
      const samples: number[] = [];
      let firstIds: string[] = [];
      for (let sample = 0; sample < 5; sample++) {
        const books = catalogue(size);
        // Each sample uses a new array, so both implementations include their cold index work.
        const start = performance.now();
        const ranked = newApi
          ? engine.searchBooks(books, { ...options, limit: 16 }).items
          : engine.recommendBooks(books, { ...options, limit: size }).slice(0, 16);
        samples.push(performance.now() - start);
        firstIds = ranked.map((item: { book: Book }) => item.book.id);
      }
      samples.sort((a, b) => a - b);
      report.push({ size, scenario: name, median_ms: +samples[2].toFixed(3), samples_ms: samples.map(x => +x.toFixed(3)), first_ids: firstIds });
    }
  }
  console.log(JSON.stringify({ node: process.version, implementation: modulePath, mode: newApi ? "bounded-first-page" : "previous-ui-full-ranking", samples_per_case: 5, results: report }, null, 2));
}
void main();
