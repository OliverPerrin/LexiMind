"use client";

import { useState } from "react";
import type { Book } from "@/lib/types";
import { Icon } from "./icons";

export function BookCover({
  book,
  eager = false,
}: {
  book: Book;
  eager?: boolean;
}) {
  const [failed, setFailed] = useState(false);
  const tone =
    Array.from(book.id).reduce((sum, letter) => sum + letter.charCodeAt(0), 0) %
    5;
  return (
    <div className={`book-cover cover-tone-${tone}`}>
      {book.coverUrl && !failed ? (
        // Source covers are cached independently of Next's image service; fallback remains usable offline.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={book.coverUrl}
          alt=""
          loading={eager ? "eager" : "lazy"}
          decoding="async"
          onError={() => setFailed(true)}
        />
      ) : (
        <div className="cover-fallback" aria-hidden="true">
          <span className="cover-rule" />
          <Icon name="book" />
          <span className="cover-title">{book.title}</span>
          <span className="cover-author">{book.authors.join(" · ")}</span>
          <span className="cover-rule" />
        </div>
      )}
    </div>
  );
}
