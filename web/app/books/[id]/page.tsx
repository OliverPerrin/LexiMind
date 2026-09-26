import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { books, getBook } from "@/lib/catalog";
import { recommendBooks } from "@/lib/recommendations";
import { BookCover } from "@/components/book-cover";
import { Icon } from "@/components/icons";

type Props = { params: Promise<{ id: string }> };
export const dynamicParams = false;
export function generateStaticParams() {
  return books.map((book) => ({ id: book.id }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const book = getBook((await params).id);
  if (!book) return { title: "Book not found — LexiMind" };
  return {
    title: `${book.title} — LexiMind`,
    description:
      book.description.slice(0, 160) ||
      `Discover ${book.title} by ${book.authors.join(", ")} and find your next read with LexiMind.`,
  };
}

export default async function BookPage({ params }: Props) {
  const book = getBook((await params).id);
  if (!book) notFound();
  const related = recommendBooks(books, { seedIds: [book.id], limit: 6 });
  return (
    <>
      <a className="skip-link" href="#book-details">
        Skip to book details
      </a>
      <header className="site-header page-width">
        <Link href="/" className="brand" aria-label="LexiMind home">
          <span className="brand-mark">
            <Icon name="book" />
          </span>
          LexiMind<span className="brand-period">.</span>
        </Link>
        <Link href="/" className="text-link">
          Back to discovery
          <Icon name="arrow" />
        </Link>
      </header>
      <main className="page-width book-page">
        <nav className="book-breadcrumb" aria-label="Breadcrumb">
          <Link href="/">The library</Link>
          <span aria-hidden="true">/</span>
          <span>{book.title}</span>
        </nav>
        <article className="full-book" id="book-details">
          <div className="full-book-cover">
            <BookCover book={book} eager />
            <p>A new chapter awaits.</p>
          </div>
          <div className="full-book-copy">
            <p className="eyebrow">BETWEEN THESE PAGES</p>
            <h1>{book.title}</h1>
            <p className="full-book-author">
              {book.authors.join(", ")}
              {book.firstPublished ? (
                <span> · First published {book.firstPublished}</span>
              ) : null}
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
              <section className="dialog-subjects">
                <h2>Follow a thread</h2>
                <div className="subject-tags">
                  {book.subjects.slice(0, 12).map((subject) => (
                    <Link
                      key={subject}
                      href={`/?subject=${encodeURIComponent(subject)}`}
                      aria-label={`Browse books about ${subject}`}
                    >
                      {subject}
                    </Link>
                  ))}
                </div>
              </section>
            )}
            <div className="dialog-actions">
              <Link
                className="button button-primary"
                href={`/?book=${book.id}`}
              >
                <Icon name="bookmark" width="17" height="17" />
                Open in discovery
              </Link>
              <Link
                className="button button-secondary"
                href={`/?similar=${book.id}`}
              >
                Find similar books
                <Icon name="arrow" width="17" height="17" />
              </Link>
            </div>
            <p className="full-book-shelf-note">
              Save it to your reading list or mark it as a favourite in
              discovery.
            </p>
            <div className="source-note">
              <a
                href={book.source.url}
                target="_blank"
                rel="noopener noreferrer"
              >
                View on {book.source.name}
                <Icon name="external" width="13" height="13" />
              </a>
              <p>
                Book details from {book.source.name}. Retrieved{" "}
                {new Intl.DateTimeFormat("en-GB", {
                  dateStyle: "long",
                  timeZone: "UTC",
                }).format(new Date(book.source.retrievedAt))}
                .
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
          </div>
        </article>
        {related.length > 0 && (
          <section className="related-books" aria-labelledby="related-heading">
            <p className="eyebrow">KEEP THE STORY GOING</p>
            <h2 id="related-heading">A few more doors to open.</h2>
            <div className="book-grid">
              {related.map(({ book: other, reasons }) => (
                <article className="book-card" key={other.id}>
                  <div className="card-cover-wrap">
                    <Link
                      href={`/books/${other.id}`}
                      className="cover-button"
                      aria-label={`Explore ${other.title}`}
                    >
                      <BookCover book={other} />
                    </Link>
                  </div>
                  <div className="card-copy">
                    <p className="book-genre">
                      {other.genres[0] ?? "From the library"}
                    </p>
                    <h3>
                      <Link href={`/books/${other.id}`}>{other.title}</Link>
                    </h3>
                    <p className="book-author">{other.authors.join(", ")}</p>
                    <p className="match-reason">{reasons[0]}</p>
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}
      </main>
      <footer className="site-footer page-width">
        <span className="footer-brand">LexiMind.</span>
        <p>An independent project for curious readers.</p>
        <Link href="/about">
          Sources &amp; the story behind it
          <Icon name="arrow" width="15" height="15" />
        </Link>
      </footer>
    </>
  );
}
