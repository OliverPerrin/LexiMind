import { BookExplorer } from "@/components/book-explorer";
import { books } from "@/lib/catalog";

export default function Home() {
  return <BookExplorer books={books} />;
}
