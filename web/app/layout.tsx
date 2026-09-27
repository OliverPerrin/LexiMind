import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LexiMind — Follow your curiosity",
  description:
    "Find your next book through ideas, genres, and the books you already love. A small, thoughtfully sourced library for curious readers.",
  applicationName: "LexiMind",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
