"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type { Book } from "@/lib/types";
import { recommendBooks } from "@/lib/recommendations";
import { BookCover } from "./book-cover";
import { Icon } from "./icons";

const STORAGE_KEY = "leximind.reading-shelf.v1";
type Preferences = {
  saved: string[];
  favorites: string[];
  dismissed: string[];
};
type ShelfTab = "saved" | "favorites" | "dismissed";
const emptyPreferences: Preferences = {
  saved: [],
  favorites: [],
  dismissed: [],
};

function readPreferences(): Preferences {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    if (!value || typeof value !== "object") return emptyPreferences;
    const ids = (key: string): string[] =>
      Array.isArray(value[key])
        ? [
            ...new Set(
              value[key].filter(
                (id: unknown): id is string => typeof id === "string",
              ),
            ),
          ]
        : [];
    return {
      saved: ids("saved"),
      favorites: ids("favorites"),
      dismissed: ids("dismissed"),
    };
  } catch {
    return emptyPreferences;
  }
}

export function BookExplorer({ books }: { books: Book[] }) {
  const [view, setView] = useState<"explore" | "shelf">("explore");
  const [shelfTab, setShelfTab] = useState<ShelfTab>("saved");
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState("");
  const [genres, setGenres] = useState<string[]>([]);
  const [moods, setMoods] = useState<string[]>([]);
  const [seedId, setSeedId] = useState<string | null>(null);
  const [preferences, setPreferences] = useState<Preferences>(emptyPreferences);
  const [storageReady, setStorageReady] = useState(false);
  const [storageError, setStorageError] = useState(false);
  const [selectedBook, setSelectedBook] = useState<Book | null>(null);
  const [notice, setNotice] = useState("");
  const [pageSize, setPageSize] = useState(16);
  const resultsRef = useRef<HTMLElement>(null);

  useEffect(() => {
    // Read browser-only preferences after hydration; actions persist explicit changes.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPreferences(readPreferences());
    setStorageReady(true);
    const entry = new URLSearchParams(window.location.search);
    const linkedBook = books.find((book) => book.id === entry.get("book"));
    const linkedSeed = books.find((book) => book.id === entry.get("similar"));
    if (linkedBook) setSelectedBook(linkedBook);
    if (linkedSeed) {
      setSeedId(linkedSeed.id);
      requestAnimationFrame(() =>
        resultsRef.current?.scrollIntoView({ block: "start" }),
      );
    }
    const onStorage = (event: StorageEvent) => {
      if (event.key === STORAGE_KEY) setPreferences(readPreferences());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [books]);

  const availableGenres = useMemo(
    () => [...new Set(books.flatMap((book) => book.genres))].sort(),
    [books],
  );
  const availableMoods = useMemo(
    () => [...new Set(books.flatMap((book) => book.moods))].sort(),
    [books],
  );
  const seed = books.find((book) => book.id === seedId);
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
      recommendBooks(books, {
        query,
        genres,
        moods,
        seedIds: seedId ? [seedId] : [],
        savedIds: preferences.favorites,
        dismissedIds: preferences.dismissed,
        limit: books.length,
      }),
    [
      books,
      query,
      genres,
      moods,
      seedId,
      preferences.favorites,
      preferences.dismissed,
    ],
  );
  const shelfBooks = useMemo(
    () =>
      books
        .filter((book) => preferences[shelfTab].includes(book.id))
        .map((book) => ({ book, score: 0, reasons: [] as string[] })),
    [books, preferences, shelfTab],
  );
  const results = view === "explore" ? rankedBooks : shelfBooks;
  const activeFilters = Boolean(
    query || genres.length || moods.length || seedId,
  );
  const shelfCount = books.filter((book) =>
    preferences.saved.includes(book.id),
  ).length;

  function updatePreference(key: ShelfTab, book: Book) {
    const removing = preferences[key].includes(book.id);
    const next = {
      ...preferences,
      [key]: removing
        ? preferences[key].filter((id) => id !== book.id)
        : [...preferences[key], book.id],
    };
    if (!removing && key !== "dismissed")
      next.dismissed = next.dismissed.filter((id) => id !== book.id);
    setPreferences(next);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      setStorageError(false);
    } catch {
      setStorageError(true);
    }
    const messages: Record<ShelfTab, string[]> = {
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
  }

  function resetFilters() {
    setDraft("");
    setQuery("");
    setGenres([]);
    setMoods([]);
    setSeedId(null);
    setPageSize(16);
  }
  function explore() {
    setView("explore");
    setPageSize(16);
  }
  function scrollToResults() {
    requestAnimationFrame(() =>
      resultsRef.current?.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
        block: "start",
      }),
    );
  }
  function search(event?: FormEvent, value = draft) {
    event?.preventDefault();
    setDraft(value);
    setQuery(value.trim());
    setView("explore");
    setPageSize(16);
    scrollToResults();
  }
  function moreLike(book: Book) {
    resetFilters();
    setSeedId(book.id);
    setView("explore");
    setSelectedBook(null);
    scrollToResults();
    setNotice(`Showing books related to ${book.title}.`);
  }
  function toggleFilter(value: string, kind: "genre" | "mood") {
    const setter = kind === "genre" ? setGenres : setMoods;
    setter((selected) =>
      selected.includes(value)
        ? selected.filter((item) => item !== value)
        : [...selected, value],
    );
    setPageSize(16);
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
              <h2 id="collection-title">
                {view === "shelf"
                  ? "Your reading life."
                  : seed
                    ? "Turn to something similar."
                    : activeFilters
                      ? "A little closer to your next read."
                      : "There’s a story for you here."}
              </h2>
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
              <div
                className="shelf-tabs"
                role="group"
                aria-label="Shelf sections"
              >
                {(["saved", "favorites", "dismissed"] as ShelfTab[]).map(
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
                      <span>
                        {
                          books.filter((book) =>
                            preferences[tab].includes(book.id),
                          ).length
                        }
                      </span>
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
                  {availableGenres.map((genre) => (
                    <button
                      key={genre}
                      className={`genre-option ${genres.includes(genre) ? "selected" : ""}`}
                      onClick={() => toggleFilter(genre, "genre")}
                      aria-pressed={genres.includes(genre)}
                    >
                      {genre}
                      <span>
                        {
                          books.filter((book) => book.genres.includes(genre))
                            .length
                        }
                      </span>
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
              {(seed || query || genres.length > 0 || moods.length > 0) &&
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
                      >
                        {genre}
                        <Icon name="close" width="14" height="14" />
                      </button>
                    ))}
                    {moods.map((mood) => (
                      <button
                        key={mood}
                        className="active-chip"
                        onClick={() => toggleFilter(mood, "mood")}
                      >
                        {mood}
                        <Icon name="close" width="14" height="14" />
                      </button>
                    ))}
                  </div>
                )}
              <div className="results-heading">
                <p role="status">
                  {results.length} {results.length === 1 ? "book" : "books"}
                  {view === "shelf"
                    ? " on this shelf"
                    : activeFilters
                      ? " found"
                      : " to get lost in"}
                </p>
                <span>
                  {view === "explore" &&
                    (preferences.favorites.length > 0
                      ? "Shaped by your favourites"
                      : activeFilters
                        ? "Sorted by relevance"
                        : "A place to begin")}
                </span>
              </div>
              {results.length === 0 ? (
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
                  {results.slice(0, pageSize).map(({ book, reasons }) => (
                    <article key={book.id} className="book-card">
                      <div className="card-cover-wrap">
                        <button
                          className="cover-button"
                          aria-label={`View ${book.title}`}
                          onClick={() => setSelectedBook(book)}
                        >
                          <BookCover book={book} />
                        </button>
                        <button
                          className={`favorite-button ${preferences.favorites.includes(book.id) ? "is-favorite" : ""}`}
                          aria-label={`${preferences.favorites.includes(book.id) ? "Remove" : "Add"} ${book.title} ${preferences.favorites.includes(book.id) ? "from" : "to"} favourites`}
                          aria-pressed={preferences.favorites.includes(book.id)}
                          disabled={!storageReady}
                          onClick={() => updatePreference("favorites", book)}
                        >
                          <Icon name="heart" width="17" height="17" />
                        </button>
                      </div>
                      <div className="card-copy">
                        <p className="book-genre">
                          {book.genres[0] ?? "From the library"}
                        </p>
                        <h3>
                          <button onClick={() => setSelectedBook(book)}>
                            {book.title}
                          </button>
                        </h3>
                        <p className="book-author">{book.authors.join(", ")}</p>
                        {reasons.length > 0 && (
                          <p className="match-reason">{reasons[0]}</p>
                        )}
                        <div className="card-actions">
                          <button
                            className={`save-button ${preferences.saved.includes(book.id) ? "is-saved" : ""}`}
                            disabled={!storageReady}
                            onClick={() => updatePreference("saved", book)}
                            aria-label={`${preferences.saved.includes(book.id) ? "Unsave" : "Save"} ${book.title}`}
                            aria-pressed={preferences.saved.includes(book.id)}
                          >
                            <Icon
                              name={
                                preferences.saved.includes(book.id)
                                  ? "check"
                                  : "bookmark"
                              }
                              width="15"
                              height="15"
                            />
                            {preferences.saved.includes(book.id)
                              ? "Saved"
                              : "Save"}
                          </button>
                          {view === "shelf" && shelfTab === "dismissed" ? (
                            <button
                              className="similar-button"
                              onClick={() =>
                                updatePreference("dismissed", book)
                              }
                            >
                              Restore
                              <Icon name="eye" width="14" height="14" />
                            </button>
                          ) : (
                            <button
                              className="similar-button"
                              onClick={() => moreLike(book)}
                              aria-label={`More like ${book.title}`}
                            >
                              More like this
                              <Icon name="arrow" width="14" height="14" />
                            </button>
                          )}
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              )}
              {results.length > pageSize && (
                <div className="load-more">
                  <p>
                    You’ve explored {Math.min(pageSize, results.length)} of{" "}
                    {results.length} books.
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
          Your browser couldn’t save your shelf. Changes are available for this
          visit only.
        </div>
      )}
      {selectedBook && (
        <BookDialog
          book={selectedBook}
          saved={preferences.saved.includes(selectedBook.id)}
          favorite={preferences.favorites.includes(selectedBook.id)}
          dismissed={preferences.dismissed.includes(selectedBook.id)}
          storageReady={storageReady}
          onClose={() => setSelectedBook(null)}
          onSave={() => updatePreference("saved", selectedBook)}
          onFavorite={() => updatePreference("favorites", selectedBook)}
          onHide={() => {
            updatePreference("dismissed", selectedBook);
            setSelectedBook(null);
          }}
          onSimilar={() => moreLike(selectedBook)}
        />
      )}
    </>
  );
}

function BookDialog({
  book,
  saved,
  favorite,
  dismissed,
  storageReady,
  onClose,
  onSave,
  onFavorite,
  onHide,
  onSimilar,
}: {
  book: Book;
  saved: boolean;
  favorite: boolean;
  dismissed: boolean;
  storageReady: boolean;
  onClose: () => void;
  onSave: () => void;
  onFavorite: () => void;
  onHide: () => void;
  onSimilar: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = dialogRef.current;
    const opener =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    dialog?.showModal();
    return () => {
      dialog?.close();
      if (opener?.isConnected) opener.focus({ preventScroll: true });
    };
  }, []);
  return (
    <dialog
      className="book-dialog"
      ref={dialogRef}
      aria-labelledby="dialog-title"
      onClose={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="dialog-inner">
        <button
          className="dialog-close"
          onClick={onClose}
          aria-label="Close book details"
        >
          <Icon name="close" />
        </button>
        <div className="dialog-cover-column">
          <BookCover book={book} eager />
          <button
            className={`button button-secondary favorite-detail ${favorite ? "is-favorite" : ""}`}
            onClick={onFavorite}
            disabled={!storageReady}
            aria-pressed={favorite}
          >
            <Icon name="heart" width="17" height="17" />
            {favorite ? "A favourite" : "Mark as a favourite"}
          </button>
        </div>
        <div className="dialog-copy">
          <p className="eyebrow">A NEW CHAPTER AWAITS</p>
          <h2 id="dialog-title">{book.title}</h2>
          <p className="dialog-author">
            {book.authors.join(", ")}
            {book.firstPublished ? <span> · {book.firstPublished}</span> : null}
          </p>
          <div className="book-tags">
            {book.genres.map((genre) => (
              <span key={genre}>{genre}</span>
            ))}
          </div>
          <p className="dialog-description">
            {book.description ||
              "A description isn’t available for this book yet. Visit the source record to learn more."}
          </p>
          {book.subjects.length > 0 && (
            <div className="dialog-subjects">
              <h3>Between these pages</h3>
              <p>{book.subjects.slice(0, 8).join(" · ")}</p>
            </div>
          )}
          <div className="dialog-actions">
            <button
              className="button button-primary"
              onClick={onSave}
              disabled={!storageReady}
              aria-pressed={saved}
            >
              <Icon
                name={saved ? "check" : "bookmark"}
                width="17"
                height="17"
              />
              {saved ? "Saved to reading list" : "Save to reading list"}
            </button>
            <button className="button button-secondary" onClick={onSimilar}>
              More like this
              <Icon name="arrow" width="17" height="17" />
            </button>
          </div>
          <div className="source-note">
            <a href={book.source.url} target="_blank" rel="noopener noreferrer">
              View on {book.source.name}
              <Icon name="external" width="13" height="13" />
            </a>
            <Link className="book-permalink" href={`/books/${book.id}`}>
              Open book page
              <Icon name="arrow" width="13" height="13" />
            </Link>
            <p>
              Book details from {book.source.name}.
              {book.descriptionSource &&
              book.descriptionSource !== book.source.url &&
              /^https:\/\//.test(book.descriptionSource) ? (
                <>
                  {" "}
                  <a
                    href={book.descriptionSource}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Description source
                  </a>
                  .
                </>
              ) : (
                ""
              )}
            </p>
          </div>
          <button
            className="quiet-link hide-book"
            onClick={onHide}
            disabled={!storageReady}
          >
            {dismissed
              ? "Restore to discovery"
              : "Hide this book from discovery"}
          </button>
        </div>
      </div>
    </dialog>
  );
}
