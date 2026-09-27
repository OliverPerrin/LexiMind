import test from "node:test";
import assert from "node:assert/strict";
import {
  emptyShelf,
  exportShelf,
  MAX_SHELF_BYTES,
  MAX_SHELF_ENTRIES,
  mergeShelves,
  parseShelfImport,
  readStoredShelf,
  setShelfPreference,
  SHELF_STORAGE_KEY,
  shelfIds,
  validateShelf,
} from "./shelf";

const sample = {
  saved: ["OL123W", "OL999999W"],
  favorites: ["OL123W"],
  dismissed: ["OL456W"],
};
test("versioned shelf exports round-trip every preference including unavailable IDs", () => {
  const text = exportShelf(sample, new Date("2026-09-26T12:00:00Z"));
  assert.deepEqual(parseShelfImport(text), sample);
  assert.deepEqual(shelfIds(sample), ["OL123W", "OL999999W", "OL456W"]);
});
test("imports reject malformed, wrong-version, invalid-identity and oversized data", () => {
  const valid = JSON.parse(exportShelf(sample));
  for (const text of [
    "not json",
    "null",
    JSON.stringify(sample),
    JSON.stringify({ ...valid, version: 2 }),
    JSON.stringify({ ...valid, exportedAt: "bad date" }),
    JSON.stringify({
      ...valid,
      preferences: { ...sample, saved: ["javascript:alert(1)"] },
    }),
    JSON.stringify({ ...valid, preferences: { ...sample, favorites: "OL1W" } }),
    " ".repeat(MAX_SHELF_BYTES + 1),
  ]) {
    assert.throws(() => parseShelfImport(text));
  }
  assert.deepEqual(sample.saved, ["OL123W", "OL999999W"]);
});
test("duplicate IDs are deduplicated but unreasonable lists are rejected", () => {
  assert.deepEqual(
    validateShelf({ ...sample, saved: ["OL123W", "OL123W"] }).saved,
    ["OL123W"],
  );
  assert.throws(() =>
    validateShelf({
      ...sample,
      saved: Array(MAX_SHELF_ENTRIES + 1).fill("OL1W"),
    }),
  );
});
test("merge keeps current preferences and unknown IDs without hiding existing visible favourites", () => {
  const current = { saved: ["OL1W"], favorites: ["OL2W"], dismissed: ["OL3W"] };
  const incoming = {
    saved: ["OL4W", "OL3W"],
    favorites: ["OL5W"],
    dismissed: ["OL1W", "OL2W", "OL9W"],
  };
  const result = mergeShelves(current, incoming);
  assert.deepEqual(result.preferences, {
    saved: ["OL1W", "OL4W", "OL3W"],
    favorites: ["OL2W", "OL5W"],
    dismissed: ["OL3W", "OL9W"],
  });
  assert.deepEqual(result.added, { saved: 2, favorites: 1, dismissed: 1 });
  assert.equal(result.keptVisible, 2);
  assert.deepEqual(current.saved, ["OL1W"]);
});
test("merging twice is idempotent and over-capacity merge fails without truncation", () => {
  const once = mergeShelves(emptyShelf(), sample).preferences;
  assert.deepEqual(mergeShelves(once, sample).preferences, once);
  const full = {
    ...emptyShelf(),
    saved: Array.from({ length: MAX_SHELF_ENTRIES }, (_, i) => `OL${i}W`),
  };
  assert.throws(() =>
    mergeShelves(full, { ...emptyShelf(), saved: ["OL999999W"] }),
  );
  assert.equal(full.saved.length, MAX_SHELF_ENTRIES);
});
test("explicit preference actions unhide on save or favourite and retain unrelated array identities", () => {
  const start = { saved: ["OL1W"], favorites: ["OL2W"], dismissed: ["OL3W"] };
  const added = setShelfPreference(start, "saved", "OL4W", true);
  assert.equal(added.favorites, start.favorites);
  assert.equal(added.dismissed, start.dismissed);
  assert.deepEqual(
    setShelfPreference(start, "favorites", "OL3W", true).dismissed,
    [],
  );
  const hidden = setShelfPreference(start, "dismissed", "OL1W", true);
  assert.deepEqual(hidden.saved, ["OL1W"]);
  assert.deepEqual(hidden.dismissed, ["OL3W", "OL1W"]);
  assert.deepEqual(setShelfPreference(start, "saved", "OL1W", false).saved, []);
});
test("storage distinguishes missing, blocked and corrupt values without mutating storage", () => {
  assert.equal(readStoredShelf(null).status, "unavailable");
  assert.equal(
    readStoredShelf({
      getItem: () => {
        throw new Error("blocked");
      },
    }).status,
    "unavailable",
  );
  assert.equal(readStoredShelf({ getItem: () => null }).status, "empty");
  assert.equal(readStoredShelf({ getItem: () => "corrupt" }).status, "invalid");
  assert.deepEqual(
    readStoredShelf({
      getItem: (key) => {
        assert.equal(key, SHELF_STORAGE_KEY);
        return JSON.stringify(sample);
      },
    }),
    { status: "ok", preferences: sample },
  );
});

test("rapid queued updates preserve both actions and read another tab's latest changes", async () => {
  const { createShelfStore } = await import("./shelf");
  let raw: string | null = null;
  const storage = {
    getItem: () => raw,
    setItem: (_key: string, value: string) => {
      raw = value;
    },
  };
  const first = createShelfStore({ storage: () => storage });
  const second = createShelfStore({ storage: () => storage });
  await Promise.all([
    first.update((shelf) => setShelfPreference(shelf, "saved", "OL1W", true)),
    first.update((shelf) => setShelfPreference(shelf, "saved", "OL2W", true)),
    second.update((shelf) =>
      setShelfPreference(shelf, "favorites", "OL3W", true),
    ),
  ]);
  first.refresh();
  assert.deepEqual(first.getSnapshot().preferences, {
    saved: ["OL1W", "OL2W"],
    favorites: ["OL3W"],
    dismissed: [],
  });
  raw = null;
  first.refresh();
  assert.deepEqual(first.getSnapshot().preferences, emptyShelf());
});
test("failed writes remain available and replay on latest remote data when storage recovers", async () => {
  const { createShelfStore } = await import("./shelf");
  let raw = JSON.stringify({ ...emptyShelf(), saved: ["OL1W"] });
  let blocked = true;
  const store = createShelfStore({
    storage: () => ({
      getItem: () => raw,
      setItem: (_key, value) => {
        if (blocked) throw new Error("quota");
        raw = value;
      },
    }),
  });
  await store.update((shelf) =>
    setShelfPreference(shelf, "saved", "OL1W", false),
  );
  await store.update((shelf) =>
    setShelfPreference(shelf, "saved", "OL2W", true),
  );
  assert.deepEqual(store.getSnapshot().preferences.saved, ["OL2W"]);
  assert.equal(store.getSnapshot().error, "write");
  raw = JSON.stringify({ ...emptyShelf(), saved: ["OL1W", "OL3W"] });
  blocked = false;
  await store.update((shelf) =>
    setShelfPreference(shelf, "favorites", "OL4W", true),
  );
  assert.deepEqual(JSON.parse(raw), {
    saved: ["OL3W", "OL2W"],
    favorites: ["OL4W"],
    dismissed: [],
  });
  assert.equal(store.getSnapshot().error, null);
});
test("corrupt storage is retained while current-visit actions remain exportable", async () => {
  const { createShelfStore } = await import("./shelf");
  let raw = "corrupt original";
  const store = createShelfStore({
    storage: () => ({
      getItem: () => raw,
      setItem: (_key, value) => {
        raw = value;
      },
    }),
  });
  await store.update((shelf) =>
    setShelfPreference(shelf, "saved", "OL1W", true),
  );
  assert.equal(raw, "corrupt original");
  assert.equal(store.getSnapshot().error, "invalid");
  assert.deepEqual(
    parseShelfImport(exportShelf(store.getSnapshot().preferences)).saved,
    ["OL1W"],
  );
});
test("rejected operations do not poison the queue or partially mutate stored preferences", async () => {
  const { createShelfStore } = await import("./shelf");
  let raw: string | null = null;
  const store = createShelfStore({
    storage: () => ({
      getItem: () => raw,
      setItem: (_key, value) => {
        raw = value;
      },
    }),
  });
  await assert.rejects(
    store.update(() => {
      throw new Error("invalid import");
    }),
  );
  assert.equal(raw, null);
  await store.update((shelf) =>
    setShelfPreference(shelf, "saved", "OL1W", true),
  );
  assert.deepEqual(store.getSnapshot().preferences.saved, ["OL1W"]);
});

test("unchanged storage notifications and unrelated shelf actions keep snapshot identities stable", async () => {
  const { createShelfStore } = await import("./shelf");
  let raw: string | null = null;
  const store = createShelfStore({
    storage: () => ({
      getItem: () => raw,
      setItem: (_key, value) => {
        raw = value;
      },
    }),
  });
  await store.update((shelf) =>
    setShelfPreference(shelf, "favorites", "OL1W", true),
  );
  const favoriteIds = store.getSnapshot().preferences.favorites;
  await store.update((shelf) =>
    setShelfPreference(shelf, "saved", "OL2W", true),
  );
  assert.equal(store.getSnapshot().preferences.favorites, favoriteIds);
  const snapshot = store.getSnapshot();
  store.refresh();
  assert.equal(store.getSnapshot(), snapshot);
});
