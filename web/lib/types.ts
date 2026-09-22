export interface Book {
  id: string;
  title: string;
  authors: string[];
  description: string;
  coverUrl: string | null;
  genres: string[];
  subjects: string[];
  /** Only independently sourced labels belong here, never sampled predictions. */
  moods: string[];
  firstPublished: number | null;
  source: { name: string; url: string; retrievedAt: string };
  identifiers: { openLibraryWork: string; isbns: string[] };
  descriptionSource?: string | null;
}

export interface RecommendationOptions {
  query?: string;
  genres?: string[];
  moods?: string[];
  seedIds?: string[];
  savedIds?: string[];
  dismissedIds?: string[];
  limit?: number;
}

export interface Recommendation {
  book: Book;
  /** Relative ranking score; not a probability, rating, or quality assessment. */
  score: number;
  reasons: string[];
}
