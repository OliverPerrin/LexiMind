"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
import type { Book } from "@/lib/types";
import { BookCover } from "./book-cover";
import { Icon } from "./icons";

export function BookDialog({
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
  onSubject,
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
  onSubject: (subject: string) => void;
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
      else
        document.getElementById("collection")?.focus({ preventScroll: true });
    };
  }, []);
  return (
    <dialog
      className="book-dialog"
      ref={dialogRef}
      aria-labelledby="dialog-title"
      onClose={() => {
        // Strict Mode may reopen after effect cleanup before its queued close event.
        if (dialogRef.current && !dialogRef.current.open) onClose();
      }}
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
              <div className="subject-tags">
                {book.subjects.slice(0, 12).map((subject) => (
                  <a
                    key={subject}
                    href={`/?subject=${encodeURIComponent(subject)}`}
                    onClick={(event) => {
                      if (
                        !event.metaKey &&
                        !event.ctrlKey &&
                        !event.shiftKey &&
                        !event.altKey
                      ) {
                        event.preventDefault();
                        onSubject(subject);
                      }
                    }}
                    aria-label={`Browse books about ${subject}`}
                  >
                    {subject}
                  </a>
                ))}
              </div>
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
