"use client";

import { memo } from "react";
import type { Book } from "@/lib/types";
import type { ShelfKey } from "@/lib/shelf";
import { BookCover } from "./book-cover";
import { Icon } from "./icons";

export const BookCard = memo(function BookCard({
  book,
  reason,
  saved,
  favorite,
  restore,
  ready,
  onOpen,
  onPreference,
  onSimilar,
}: {
  book: Book;
  reason: string;
  saved: boolean;
  favorite: boolean;
  restore: boolean;
  ready: boolean;
  onOpen: (book: Book) => void;
  onPreference: (key: ShelfKey, book: Book) => void;
  onSimilar: (book: Book) => void;
}) {
  return (
    <article className="book-card">
      <div className="card-cover-wrap">
        <button
          className="cover-button"
          aria-label={`View ${book.title}`}
          onClick={() => onOpen(book)}
        >
          <BookCover book={book} />
        </button>
        <button
          className={`favorite-button ${favorite ? "is-favorite" : ""}`}
          aria-label={`${favorite ? "Remove" : "Add"} ${book.title} ${favorite ? "from" : "to"} favourites`}
          aria-pressed={favorite}
          disabled={!ready}
          onClick={() => onPreference("favorites", book)}
        >
          <Icon name="heart" width="17" height="17" />
        </button>
      </div>
      <div className="card-copy">
        <p className="book-genre">{book.genres[0] ?? "From the library"}</p>
        <h3>
          <button onClick={() => onOpen(book)}>{book.title}</button>
        </h3>
        <p className="book-author">{book.authors.join(", ")}</p>
        {reason && <p className="match-reason">{reason}</p>}
        <div className="card-actions">
          <button
            className={`save-button ${saved ? "is-saved" : ""}`}
            disabled={!ready}
            onClick={() => onPreference("saved", book)}
            aria-label={`${saved ? "Unsave" : "Save"} ${book.title}`}
            aria-pressed={saved}
          >
            <Icon name={saved ? "check" : "bookmark"} width="15" height="15" />
            {saved ? "Saved" : "Save"}
          </button>
          {restore ? (
            <button
              className="similar-button"
              disabled={!ready}
              onClick={() => onPreference("dismissed", book)}
            >
              Restore
              <Icon name="eye" width="14" height="14" />
            </button>
          ) : (
            <button
              className="similar-button"
              onClick={() => onSimilar(book)}
              aria-label={`More like ${book.title}`}
            >
              More like this
              <Icon name="arrow" width="14" height="14" />
            </button>
          )}
        </div>
      </div>
    </article>
  );
});
