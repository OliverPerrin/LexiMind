import Link from "next/link";
import { Icon } from "@/components/icons";

export default function NotFound() {
  return (
    <main className="page-width about-main">
      <p className="eyebrow">A PAGE OUT OF PLACE</p>
      <h1>
        This chapter
        <br />
        <em>is missing.</em>
      </h1>
      <p>
        That book or page isn’t in this catalogue. There are plenty of other
        stories waiting on the shelf.
      </p>
      <Link href="/" className="button button-primary">
        Back to discovery
        <Icon name="arrow" />
      </Link>
    </main>
  );
}
