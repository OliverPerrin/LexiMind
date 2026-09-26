"use client";

import { useMemo, useRef, useState, type ChangeEvent } from "react";
import {
  exportShelf,
  MAX_SHELF_BYTES,
  parseShelfImport,
  shelfIds,
  type ShelfPreferences,
  type mergeShelves,
} from "@/lib/shelf";

type ImportReport = ReturnType<typeof mergeShelves>;
export function ShelfTransfer({
  preferences,
  knownIds,
  ready,
  onImport,
}: {
  preferences: ShelfPreferences;
  knownIds: ReadonlySet<string>;
  ready: boolean;
  onImport: (incoming: ShelfPreferences) => Promise<ImportReport>;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const unavailable = useMemo(
    () => shelfIds(preferences).filter((id) => !knownIds.has(id)),
    [preferences, knownIds],
  );

  function download() {
    setError("");
    try {
      const url = URL.createObjectURL(
        new Blob([exportShelf(preferences)], { type: "application/json" }),
      );
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `leximind-shelf-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
      setMessage(
        "Your shelf export is ready to download. Keep the file to restore it in another browser.",
      );
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "The shelf could not be exported. Your saved books have not changed.",
      );
    }
  }

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      if (file.size > MAX_SHELF_BYTES)
        throw new Error(
          "This shelf file is too large. The maximum size is 256 KB.",
        );
      const incoming = parseShelfImport(await file.text());
      const report = await onImport(incoming);
      const unknownCount = shelfIds(incoming).filter(
        (id) => !knownIds.has(id),
      ).length;
      setMessage(
        `Merged ${report.added.saved} reading-list, ${report.added.favorites} favourite, and ${report.added.dismissed} hidden entries. Your existing shelf was kept.${report.keptVisible ? ` ${report.keptVisible} existing books were kept visible.` : ""}${unknownCount ? ` ${unknownCount} book ${unknownCount === 1 ? "ID is" : "IDs are"} not in this catalogue yet; they are kept in your shelf and exports.` : ""}`,
      );
    } catch (cause) {
      setError(
        `${cause instanceof Error ? cause.message : "This file could not be imported."} Nothing was removed from your shelf.`,
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="shelf-transfer" aria-label="Shelf backup and import">
      <details>
        <summary>Back up or move your shelf</summary>
        <p>
          Export your reading list, favourites, and hidden books as a small JSON
          file. Importing adds to this shelf; it never replaces your existing
          books. Files stay on your device.
        </p>
        <div className="shelf-transfer-actions">
          <button
            className="button button-secondary"
            disabled={!ready || busy}
            onClick={download}
          >
            Export shelf
          </button>
          <button
            className="button button-secondary"
            disabled={!ready || busy}
            onClick={() => fileInput.current?.click()}
          >
            {busy ? "Importing…" : "Import shelf"}
          </button>
          <label className="sr-only" htmlFor="shelf-import">
            Choose a shelf export
          </label>
          <input
            ref={fileInput}
            id="shelf-import"
            className="sr-only"
            type="file"
            accept=".json,application/json"
            onChange={upload}
            tabIndex={-1}
            disabled={!ready || busy}
          />
        </div>
        <p className="shelf-file-note">
          LexiMind JSON export · Up to 256 KB · At most 2,000 entries per list
        </p>
      </details>
      {message && (
        <p className="transfer-message" role="status">
          {message}
        </p>
      )}
      {error && (
        <p className="transfer-error" role="alert">
          {error}
        </p>
      )}
      {unavailable.length > 0 && (
        <details className="unavailable-books">
          <summary>
            {unavailable.length}{" "}
            {unavailable.length === 1 ? "book is" : "books are"} outside this
            catalogue
          </summary>
          <p>
            Their work IDs are safely kept in your shelf and exports. They will
            appear when the catalogue includes them.
          </p>
          <ul>
            {unavailable.slice(0, 20).map((id) => (
              <li key={id}>{id}</li>
            ))}
          </ul>
          {unavailable.length > 20 && (
            <p>And {unavailable.length - 20} more IDs in your export.</p>
          )}
        </details>
      )}
    </section>
  );
}
