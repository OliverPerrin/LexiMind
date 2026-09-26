"use client";

import Link from "next/link";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import type { Book } from "@/lib/types";
import { searchBooks } from "@/lib/recommendations";
import { type ShelfKey } from "@/lib/shelf";
import { useReadingShelf } from "./use-reading-shelf";
import { ShelfTransfer } from "./shelf-transfer";
import { BookDialog } from "./book-dialog";
import { BookCard } from "./book-card";
import { BookCover } from "./book-cover";
import { Icon } from "./icons";

export function BookExplorer({ books }: { books: Book[] }) {
  const [view, setView] = useState<"explore" | "shelf">("explore");
  const [shelfTab, setShelfTab] = useState<ShelfKey>("saved");
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState("");
  const [genres, setGenres] = useState<string[]>([]);
  const [moods, setMoods] = useState<string[]>([]);
  const [subjects, setSubjects] = useState<string[]>([]);
  const [seedId, setSeedId] = useState<string | null>(null);
  const {
    preferences,
    ready: storageReady,
    error: storageError,
    toggle,
    importPreferences,
  } = useReadingShelf();
  const [selectedBook, setSelectedBook] = useState<Book | null>(null);
  const [notice, setNotice] = useState("");
  const [pageSize, setPageSize] = useState(16);
  const resultsRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const entry = new URLSearchParams(window.location.search);
    const linkedBook = books.find((book) => book.id === entry.get("book"));
    const linkedSeed = books.find((book) => book.id === entry.get("similar"));
    // Apply source-linked discovery entry points after hydration.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (linkedBook) setSelectedBook(linkedBook);
    const linkedSubjects = entry
      .getAll("subject")
      .filter((value) => value.trim() && value.length <= 300)
      .slice(0, 8);
    if (linkedSubjects.length) setSubjects(linkedSubjects);
    if (linkedSeed || linkedSubjects.length) {
      if (linkedSeed) setSeedId(linkedSeed.id);
      requestAnimationFrame(() =>
        resultsRef.current?.scrollIntoView({ block: "start" }),
      );
    }
  }, [books]);

  const catalogueIndex = useMemo(() => {
    const byId = new Map(books.map((book) => [book.id, book]));
    const genreCounts = new Map<string, number>();
    const subjectCounts = new Map<string, number>();
    const moodSet = new Set<string>();
    for (const book of books) {
      for (const genre of new Set(book.genres))
        genreCounts.set(genre, (genreCounts.get(genre) ?? 0) + 1);
      for (const subject of new Set(book.subjects))
        subjectCounts.set(subject, (subjectCounts.get(subject) ?? 0) + 1);
      for (const mood of book.moods) moodSet.add(mood);
    }
    return {
      byId,
      knownIds: new Set(byId.keys()),
      genres: [...genreCounts].sort(([a], [b]) => a.localeCompare(b)),
      subjects: [...subjectCounts]
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
        .slice(0, 50),
      moods: [...moodSet].sort(),
    };
  }, [books]);
  const availableMoods = catalogueIndex.moods;
  const seed = seedId ? catalogueIndex.byId.get(seedId) : undefined;
  const shelfIndex = useMemo(
    () => ({
      saved: new Set(preferences.saved),
      favorites: new Set(preferences.favorites),
      dismissed: new Set(preferences.dismissed),
    }),
    [preferences],
  );
  const shelfCounts = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(shelfIndex).map(([key, ids]) => [
          key,
          [...ids].filter((id) => catalogueIndex.byId.has(id)).length,
        ]),
      ) as Record<ShelfKey, number>,
    [shelfIndex, catalogueIndex],
  );
  const featured = useMemo(() => {
    const preferred = [
      "Circe",
      "Frankenstein",
      "A short history of nearly everything",
    ];
    const picks = preferred
      .map((title) =>
        books.find((book) =>
          book.title.toLowerCase().startsWith(title.toLowerCase()),
        ),
      )
      .filter((book): book is Book => Boolean(book));
    return [...picks, ...books.filter((book) => !picks.includes(book))].slice(
      0,
      3,
    );
  }, [books]);

  const rankedBooks = useMemo(
    () =>
      view === "explore"
        ? searchBooks(books, {
            query,
            genres,
            moods,
            subjects,
            seedIds: seedId ? [seedId] : [],
            savedIds: preferences.favorites,
            dismissedIds: preferences.dismissed,
            limit: pageSize,
          })
        : { items: [], total: 0 },
    [
      view,
      books,
      query,
      genres,
      moods,
      subjects,
      pageSize,
      seedId,
      preferences.favorites,
      preferences.dismissed,
    ],
  );
  const currentShelf = preferences[shelfTab];
  const shelfBooks = useMemo(
    () =>
      view === "shelf"
        ? currentShelf.flatMap((id) => {
            const book = catalogueIndex.byId.get(id);
            return book ? [{ book, score: 0, reasons: [] as string[] }] : [];
          })
        : [],
    [view, currentShelf, catalogueIndex],
  );
  const results =
    view === "explore" ? rankedBooks.items : shelfBooks.slice(0, pageSize);
  const totalResults =
    view === "explore" ? rankedBooks.total : shelfBooks.length;
  const activeFilters = Boolean(
    query || genres.length || moods.length || subjects.length || seedId,
  );
  const shelfCount = shelfCounts.saved;

  const updatePreference = useCallback(
    async (key: ShelfKey, book: Book) => {
      try {
        const enabled = await toggle(key, book.id);
        const removing = !enabled;
        const messages: Record<ShelfKey, string[]> = {
          saved: [
            `Saved ${book.title} to your reading list.`,
            `Removed ${book.title} from your reading list.`,
          ],
          favorites: [
            `Added ${book.title} to your favourites.`,
            `Removed ${book.title} from your favourites.`,
          ],
          dismissed: [
            `Hidden ${book.title}. You can restore it from your shelf.`,
            `Restored ${book.title} to discovery.`,
          ],
        };
        setNotice(messages[key][removing ? 1 : 0]);
      } catch (cause) {
        setNotice(
          cause instanceof Error
            ? cause.message
            : "The shelf could not be updated.",
        );
      }
    },
    [toggle],
  );

  const resetFilters = useCallback(() => {
    setDraft("");
    setQuery("");
    setGenres([]);
    setMoods([]);
    setSubjects([]);
    setSeedId(null);
    setPageSize(16);
  }, []);
  function explore() {
    setView("explore");
    setPageSize(16);
  }
  const scrollToResults = useCallback(() => {
    requestAnimationFrame(() =>
      resultsRef.current?.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
        block: "start",
      }),
    );
  }, []);
  function search(event?: FormEvent, value = draft) {
    event?.preventDefault();
    setDraft(value);
    setQuery(value.trim());
    setView("explore");
    setPageSize(16);
    scrollToResults();
  }
  const moreLike = useCallback(
    (book: Book) => {
      resetFilters();
      setSeedId(book.id);
      setView("explore");
      setSelectedBook(null);
      scrollToResults();
      setNotice(`Showing books related to ${book.title}.`);
    },
    [resetFilters, scrollToResults],
  );
  function toggleFilter(value: string, kind: "genre" | "mood" | "subject") {
    const setter =
      kind === "genre" ? setGenres : kind === "mood" ? setMoods : setSubjects;
    setter((selected) =>
      selected.includes(value)
        ? selected.filter((item) => item !== value)
        : [...selected, value],
    );
    setPageSize(16);
  }

  const CollectionHeading = view === "shelf" ? "h1" : "h2";

  function browseSubject(subject: string) {
    resetFilters();
    setSubjects([subject]);
    setSelectedBook(null);
    setView("explore");
    scrollToResults();
  }

  return (
    <>
      <a className="skip-link" href="#collection">
        Skip to books
      </a>
      <header className="site-header page-width">
        <Link href="/" className="brand" aria-label="LexiMind home">
          <span className="brand-mark">
            <Icon name="book" />
          </span>
          LexiMind<span className="brand-period">.</span>
        </Link>
        <nav aria-label="Main navigation">
          <button
            className={`nav-link ${view === "explore" ? "active" : ""}`}
            onClick={explore}
            aria-current={view === "explore" ? "page" : undefined}
          >
            Discover
          </button>
          <button
            className={`nav-link shelf-link ${view === "shelf" ? "active" : ""}`}
            onClick={() => {
              setView("shelf");
              setPageSize(16);
              scrollToResults();
            }}
            aria-current={view === "shelf" ? "page" : undefined}
          >
            <Icon name="bookmark" />
            My shelf<span className="nav-count">{shelfCount}</span>
          </button>
          <Link className="nav-link about-nav" href="/about">
            The project
            <Icon name="external" width="14" height="14" />
          </Link>
        </nav>
      </header>

      {view === "explore" && (
        <section className="hero page-width" aria-labelledby="hero-title">
          <div className="hero-copy">
            <p className="eyebrow">
              <span className="tiny-star">✳</span> FOR THE CURIOUS READER
            </p>
            <h1 id="hero-title">
              Follow your
              <br />
              <em>curiosity.</em>
            </h1>
            <p className="hero-description">
              A familiar favourite. A different perspective.
              <br className="desktop-break" /> A world you haven’t wandered into
              yet.
              <br className="desktop-break" /> Find a book that takes you
              somewhere.
            </p>
            <a className="text-link hero-link" href="#collection">
              Find your next read
              <Icon name="arrow" />
            </a>
          </div>
          <div className="hero-library">
            <div className="library-orbit" aria-hidden="true" />
            <span className="library-note">
              A little wonder
              <br />
              between the covers.
            </span>
            <div className="featured-books">
              {featured.map((book, index) => (
                <button
                  key={book.id}
                  className={`featured-book featured-${index}`}
                  onClick={() => setSelectedBook(book)}
                  aria-label={`Explore ${book.title}`}
                >
                  <BookCover book={book} eager />
                </button>
              ))}
            </div>
            <span className="hero-shelf-line" aria-hidden="true" />
            <span className="library-caption">
              MANY STORIES. YOUR NEXT CHAPTER.
            </span>
          </div>
        </section>
      )}

      <main className="page-width">
        {view === "explore" && (
          <section className="search-section" aria-label="Book search">
            <div className="search-heading">
              <span className="eyebrow">START WITH A LITTLE CURIOSITY</span>
              <span className="search-count">
                {books.length} books to explore
              </span>
            </div>
            <form className="search-form" onSubmit={search} role="search">
              <Icon name="search" width="24" height="24" />
              <label className="sr-only" htmlFor="book-search">
                Search by title, author, or what you want to read
              </label>
              <input
                id="book-search"
                type="search"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="Search by title, author, or what you want to read…"
                autoComplete="off"
              />
              <button
                className="button button-primary search-submit"
                type="submit"
              >
                Find a book
                <Icon name="arrow" />
              </button>
            </form>
            <div className="search-examples">
              <span>Try a new direction</span>
              {["Coming of age", "Science fiction", "Nature and adventure"].map(
                (example) => (
                  <button
                    key={example}
                    onClick={() => search(undefined, example)}
                  >
                    {example}
                    <span aria-hidden="true">↗</span>
                  </button>
                ),
              )}
            </div>
          </section>
        )}

        <section
          id="collection"
          ref={resultsRef}
          className={`collection ${view === "shelf" ? "shelf-collection" : ""}`}
          aria-labelledby="collection-title"
          tabIndex={-1}
        >
          <div className="collection-heading">
            <div>
              <p className="eyebrow">
                {view === "shelf"
                  ? "GOOD THINGS, KEPT CLOSE"
                  : seed
                    ? "KEEP THE STORY GOING"
                    : activeFilters
                      ? "FOLLOW THAT THREAD"
                      : "THE OPEN SHELF"}
              </p>
              <CollectionHeading
                id="collection-title"
                className="collection-title"
              >
                {view === "shelf"
                  ? "Your reading life."
                  : seed
                    ? "Turn to something similar."
                    : activeFilters
                      ? "A little closer to your next read."
                      : "There’s a story for you here."}
              </CollectionHeading>
            </div>
            {view === "explore" && (
              <p className="collection-note">
                Start anywhere.
                <br />
                See where a book takes you.
              </p>
            )}
          </div>

          {view === "shelf" && (
            <>
              <p className="shelf-intro">
                Save what catches your eye. Favourite the books you love to
                shape your recommendations.
              </p>
              <p className="browser-note">
                <Icon name="bookmark" width="15" height="15" />
                Your shelf stays in this browser. No account needed; it won’t
                sync to other devices.
              </p>
              <ShelfTransfer
                preferences={preferences}
                knownIds={catalogueIndex.knownIds}
                ready={storageReady}
                onImport={importPreferences}
              />
              <div
                className="shelf-tabs"
                role="group"
                aria-label="Shelf sections"
              >
                {(["saved", "favorites", "dismissed"] as ShelfKey[]).map(
                  (tab) => (
                    <button
                      key={tab}
                      aria-pressed={shelfTab === tab}
                      className={shelfTab === tab ? "selected" : ""}
                      onClick={() => {
                        setShelfTab(tab);
                        setPageSize(16);
                      }}
                    >
                      {tab === "saved"
                        ? "Reading list"
                        : tab === "favorites"
                          ? "Favourites"
                          : "Hidden"}
                      <span>{shelfCounts[tab]}</span>
                    </button>
                  ),
                )}
              </div>
            </>
          )}

          <div
            className={
              view === "explore" ? "collection-layout" : "shelf-layout"
            }
          >
            {view === "explore" && (
              <aside className="filters" aria-label="Filter books">
                <div className="filter-title">
                  <h3>
                    <Icon name="filter" width="17" height="17" />
                    Browse by genre
                  </h3>
                  {activeFilters && (
                    <button className="quiet-link" onClick={resetFilters}>
                      Reset
                    </button>
                  )}
                </div>
                <div className="genre-options">
                  <button
                    className={`genre-option ${genres.length === 0 ? "selected" : ""}`}
                    onClick={() => {
                      setGenres([]);
                      setPageSize(16);
                    }}
                    aria-pressed={genres.length === 0}
                  >
                    All genres<span>{books.length}</span>
                  </button>
                  {catalogueIndex.genres.map(([genre, count]) => (
                    <button
                      key={genre}
                      className={`genre-option ${genres.includes(genre) ? "selected" : ""}`}
                      onClick={() => toggleFilter(genre, "genre")}
                      aria-pressed={genres.includes(genre)}
                    >
                      {genre}
                      <span>{count}</span>
                    </button>
                  ))}
                </div>
                {availableMoods.length > 0 && (
                  <div className="mood-filter">
                    <h3>Find a feeling</h3>
                    <div className="mood-options">
                      {availableMoods.map((mood) => (
                        <button
                          key={mood}
                          className={`filter-chip ${moods.includes(mood) ? "selected" : ""}`}
                          onClick={() => toggleFilter(mood, "mood")}
                          aria-pressed={moods.includes(mood)}
                        >
                          {mood}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                <div className="subject-filter">
                  <label htmlFor="subject-filter">Follow a topic</label>
                  <select
                    id="subject-filter"
                    value=""
                    onChange={(event) => {
                      if (event.target.value)
                        toggleFilter(event.target.value, "subject");
                    }}
                  >
                    <option value="">Choose a source subject…</option>
                    {catalogueIndex.subjects.map(([subject, count]) => (
                      <option
                        key={subject}
                        value={subject}
                        disabled={subjects.includes(subject)}
                      >
                        {subject} ({count})
                      </option>
                    ))}
                  </select>
                  <p>
                    Topics come from the source records. Book details have more
                    to explore.
                  </p>
                </div>
                <div className="sidebar-note">
                  <Icon name="spark" width="23" height="23" />
                  <h4>Your taste. More possibilities.</h4>
                  <p>
                    Tap the heart on books you love. We’ll use their topics and
                    genres to help you find your next one.
                  </p>
                </div>
              </aside>
            )}
            <div className="results-area">
              {(seed ||
                query ||
                genres.length > 0 ||
                moods.length > 0 ||
                subjects.length > 0) &&
                view === "explore" && (
                  <div className="active-filters" aria-label="Active filters">
                    {seed && (
                      <button
                        className="active-chip"
                        onClick={() => setSeedId(null)}
                      >
                        More like {seed.title}
                        <Icon name="close" width="14" height="14" />
                      </button>
                    )}
                    {query && (
                      <button
                        className="active-chip"
                        onClick={() => {
                          setQuery("");
                          setDraft("");
                        }}
                      >
                        “{query}”<Icon name="close" width="14" height="14" />
                      </button>
                    )}
                    {genres.map((genre) => (
                      <button
                        key={genre}
                        className="active-chip"
                        onClick={() => toggleFilter(genre, "genre")}
                        aria-label={`Remove genre ${genre}`}
                      >
                        {genre}
                        <Icon name="close" width="14" height="14" />
                      </button>
                    ))}
                    {subjects.map((subject) => (
                      <button
                        key={subject}
                        className="active-chip"
                        onClick={() => toggleFilter(subject, "subject")}
                        aria-label={`Remove topic ${subject}`}
                      >
                        Topic: {subject}
                        <Icon name="close" width="14" height="14" />
                      </button>
                    ))}
                    {moods.map((mood) => (
                      <button
                        key={mood}
                        className="active-chip"
                        onClick={() => toggleFilter(mood, "mood")}
                        aria-label={`Remove mood ${mood}`}
                      >
                        {mood}
                        <Icon name="close" width="14" height="14" />
                      </button>
                    ))}
                  </div>
                )}
              <div className="results-heading">
                <p role="status">
                  {totalResults} {totalResults === 1 ? "book" : "books"}
                  {view === "shelf"
                    ? " on this shelf"
                    : activeFilters
                      ? " found"
                      : " to get lost in"}
                </p>
                <span>
                  {view === "explore" &&
                    (shelfCounts.favorites > 0
                      ? "Shaped by your favourites"
                      : activeFilters
                        ? "Sorted by relevance"
                        : "A place to begin")}
                </span>
              </div>
              {totalResults === 0 ? (
                <div className="empty-state">
                  <span className="empty-icon">
                    <Icon
                      name={view === "shelf" ? "bookmark" : "search"}
                      width="30"
                      height="30"
                    />
                  </span>
                  <h3>
                    {view === "shelf"
                      ? shelfTab === "dismissed"
                        ? "No books hidden away."
                        : shelfTab === "favorites"
                          ? "The books that stay with you."
                          : "A shelf full of possibilities."
                      : "No books on this path. Yet."}
                  </h3>
                  <p>
                    {view === "shelf"
                      ? shelfTab === "dismissed"
                        ? "Books you hide from discovery will appear here, ready to restore whenever you like."
                        : shelfTab === "favorites"
                          ? "Heart a book you love. Your favourites will gather here and guide what you discover next."
                          : "Save a book that catches your eye and come back to it when you’re ready for a new chapter."
                      : "This is a small, growing catalogue. Try a broader idea, a different author, or fewer filters."}
                  </p>
                  <button
                    className="button button-primary"
                    onClick={() => {
                      resetFilters();
                      explore();
                    }}
                  >
                    {view === "shelf" ? "Discover a book" : "Explore all books"}
                    <Icon name="arrow" />
                  </button>
                </div>
              ) : (
                <div className="book-grid">
                  {results.map(({ book, reasons }) => (
                    <BookCard
                      key={book.id}
                      book={book}
                      reason={reasons[0] ?? ""}
                      saved={shelfIndex.saved.has(book.id)}
                      favorite={shelfIndex.favorites.has(book.id)}
                      restore={view === "shelf" && shelfTab === "dismissed"}
                      ready={storageReady}
                      onOpen={setSelectedBook}
                      onPreference={updatePreference}
                      onSimilar={moreLike}
                    />
                  ))}
                </div>
              )}
              {totalResults > pageSize && (
                <div className="load-more">
                  <p>
                    You’ve explored {Math.min(pageSize, totalResults)} of{" "}
                    {totalResults} books.
                  </p>
                  <button
                    className="button button-secondary"
                    onClick={() => setPageSize((size) => size + 16)}
                  >
                    A few more possibilities
                    <Icon name="arrow" />
                  </button>
                </div>
              )}
            </div>
          </div>
        </section>
        <section className="bottom-note">
          <Icon name="book" width="28" height="28" />
          <div>
            <h2>A good book is just the beginning.</h2>
            <p>
              Follow a subject, rediscover an old favourite, or leave room for a
              little serendipity.
            </p>
          </div>
          <a className="text-link" href="#collection">
            Back to the shelf
            <Icon name="arrow" />
          </a>
        </section>
      </main>
      <footer className="site-footer page-width">
        <span className="footer-brand">LexiMind.</span>
        <p>An independent project for curious readers.</p>
        <Link href="/about">
          Sources &amp; the story behind it
          <Icon name="arrow" width="15" height="15" />
        </Link>
      </footer>
      <div className="toast-region" aria-live="polite" aria-atomic="true">
        {notice && (
          <div className="toast">
            <Icon name="check" width="18" height="18" />
            <span>{notice}</span>
            <button
              aria-label="Dismiss notification"
              onClick={() => setNotice("")}
            >
              <Icon name="close" width="16" height="16" />
            </button>
          </div>
        )}
      </div>
      {storageError && (
        <div className="storage-warning" role="alert">
          {storageError === "invalid"
            ? "The saved shelf could not be read, so its original data was left untouched."
            : "Your browser couldn’t save your shelf."}{" "}
          Changes are available for this visit. Export your shelf to keep a
          copy.
        </div>
      )}
      {selectedBook && (
        <BookDialog
          book={selectedBook}
          saved={shelfIndex.saved.has(selectedBook.id)}
          favorite={shelfIndex.favorites.has(selectedBook.id)}
          dismissed={shelfIndex.dismissed.has(selectedBook.id)}
          storageReady={storageReady}
          onClose={() => setSelectedBook(null)}
          onSave={() => updatePreference("saved", selectedBook)}
          onFavorite={() => updatePreference("favorites", selectedBook)}
          onHide={async () => {
            await updatePreference("dismissed", selectedBook);
            setSelectedBook(null);
          }}
          onSimilar={() => moreLike(selectedBook)}
          onSubject={browseSubject}
        />
      )}
    </>
  );
}
