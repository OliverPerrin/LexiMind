import Link from "next/link";
import type { Metadata } from "next";
import { Icon } from "@/components/icons";

export const metadata: Metadata = { title: "The project — LexiMind" };

export default function About() {
  return (
    <>
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
      <main className="page-width about-main">
        <p className="eyebrow">AN INDEPENDENT PROJECT FOR CURIOUS READERS</p>
        <h1>
          Good books.
          <br />
          <em>Open possibilities.</em>
        </h1>
        <p className="about-lede">
          LexiMind is a small place to discover books through the things that
          interest you, and the stories you already love.
        </p>
        <p>
          It started as an undergraduate research project exploring how a single
          model could learn to summarise text and recognise its topics and
          emotions. The original idea was simple: help people find a book that
          speaks to them. This library returns to that idea.
        </p>
        <h2>Follow an idea.</h2>
        <p>
          Search for a title, an author, or a subject. Browse a genre. Choose
          “More like this” on a book that catches your eye. Recommendations look
          for connections in book descriptions, subjects, and genres, with room
          for a different author or a new direction.
        </p>
        <p>
          These are suggestions from a small catalogue, not a complete map of
          what you might love. Every book links to its source so you can explore
          further.
        </p>
        <h2>A little shelf of your own.</h2>
        <p>
          Save books to a reading list, heart your favourites to shape your
          recommendations, and hide books you’re not interested in. Hidden books
          can always be restored from your shelf.
        </p>
        <p>
          Your shelf is stored only in this browser. There is no account, and it
          won’t sync between devices. Clearing your browser’s site data will
          clear your shelf.
        </p>
        <h2>Sources you can follow.</h2>
        <p>
          Book records link to{" "}
          <a
            href="https://openlibrary.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Open Library
          </a>
          . Descriptions and available covers come from the source records; when
          a cover is unavailable, a simple title card takes its place. This is
          an early, curated catalogue, and coverage will grow over time.
        </p>
        <h2>The research continues.</h2>
        <p>
          The question behind LexiMind is still interesting: can learning
          several language tasks together help a model understand books better?
          The website is a useful place to explore what better book discovery
          should feel like. The research models are not currently used to assign
          emotions or make recommendations here.
        </p>
        <Link href="/" className="button button-primary">
          Find your next read
          <Icon name="arrow" />
        </Link>
      </main>
      <footer className="site-footer page-width">
        <span className="footer-brand">LexiMind.</span>
        <p>A little curiosity goes a long way.</p>
      </footer>
    </>
  );
}
