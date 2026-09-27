/** Portable, local-only reading preferences. Unknown catalogue IDs are retained. */
export const SHELF_STORAGE_KEY = "leximind.reading-shelf.v1";
export const SHELF_FORMAT = "leximind.reading-shelf";
export const MAX_SHELF_BYTES = 256_000;
export const MAX_SHELF_ENTRIES = 2_000;
export const SHELF_KEYS = ["saved", "favorites", "dismissed"] as const;
export type ShelfKey = (typeof SHELF_KEYS)[number];
export type ShelfPreferences = Record<ShelfKey, string[]>;
export const emptyShelf = (): ShelfPreferences => ({
  saved: [],
  favorites: [],
  dismissed: [],
});
const validId = (id: unknown): id is string =>
  typeof id === "string" && /^OL\d{1,12}W$/.test(id);
const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);
const byteLength = (value: string) =>
  new TextEncoder().encode(value).byteLength;

export class ShelfDataError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ShelfDataError";
  }
}

export function validateShelf(value: unknown): ShelfPreferences {
  if (!isRecord(value))
    throw new ShelfDataError(
      "The shelf must contain reading list, favourite, and hidden book lists.",
    );
  const result = emptyShelf();
  for (const key of SHELF_KEYS) {
    const entries = value[key];
    if (
      !Array.isArray(entries) ||
      entries.length > MAX_SHELF_ENTRIES ||
      !entries.every(validId)
    ) {
      throw new ShelfDataError(
        `The ${key} list must contain no more than ${MAX_SHELF_ENTRIES.toLocaleString("en-GB")} valid Open Library work IDs.`,
      );
    }
    result[key] = [...new Set(entries)];
  }
  return result;
}

function parseJson(text: string): unknown {
  // Check length before parsing; callers also check File.size before reading an upload.
  if (text.length > MAX_SHELF_BYTES || byteLength(text) > MAX_SHELF_BYTES)
    throw new ShelfDataError(
      "This shelf file is too large. The maximum size is 256 KB.",
    );
  try {
    return JSON.parse(text);
  } catch {
    throw new ShelfDataError(
      "This file is not valid JSON. Choose a LexiMind shelf export.",
    );
  }
}

export function parseShelfImport(text: string): ShelfPreferences {
  const value = parseJson(text);
  if (!isRecord(value) || value.format !== SHELF_FORMAT)
    throw new ShelfDataError("This is not a LexiMind shelf export.");
  if (value.version !== 1)
    throw new ShelfDataError(
      "This shelf export uses an unsupported version. Your current shelf has not changed.",
    );
  if (
    typeof value.exportedAt !== "string" ||
    value.exportedAt.length > 40 ||
    !Number.isFinite(Date.parse(value.exportedAt))
  )
    throw new ShelfDataError("The shelf export has an invalid export date.");
  return validateShelf(value.preferences);
}

export function exportShelf(
  preferences: ShelfPreferences,
  now = new Date(),
): string {
  const result = JSON.stringify(
    {
      format: SHELF_FORMAT,
      version: 1,
      exportedAt: now.toISOString(),
      preferences: validateShelf(preferences),
    },
    null,
    2,
  );
  if (byteLength(result) > MAX_SHELF_BYTES)
    throw new ShelfDataError("This shelf is too large to export as one file.");
  return result;
}

export function shelfIds(preferences: ShelfPreferences): string[] {
  return [...new Set(SHELF_KEYS.flatMap((key) => preferences[key]))];
}

export function mergeShelves(
  current: ShelfPreferences,
  incoming: ShelfPreferences,
) {
  const hidden = new Set(current.dismissed);
  const visible = new Set(
    [...current.saved, ...current.favorites].filter((id) => !hidden.has(id)),
  );
  const keptVisible = incoming.dismissed.filter((id) => visible.has(id));
  const preferences = {
    saved: [...new Set([...current.saved, ...incoming.saved])],
    favorites: [...new Set([...current.favorites, ...incoming.favorites])],
    // An import cannot hide a book the reader already explicitly kept visible.
    dismissed: [
      ...new Set([
        ...current.dismissed,
        ...incoming.dismissed.filter((id) => !visible.has(id)),
      ]),
    ],
  };
  validateShelf(preferences);
  const added = Object.fromEntries(
    SHELF_KEYS.map((key) => [
      key,
      preferences[key].length - current[key].length,
    ]),
  ) as Record<ShelfKey, number>;
  return { preferences, added, keptVisible: keptVisible.length };
}

export function setShelfPreference(
  current: ShelfPreferences,
  key: ShelfKey,
  id: string,
  enabled: boolean,
): ShelfPreferences {
  if (!validId(id))
    throw new ShelfDataError("This book has an invalid catalogue identity.");
  const hasId = current[key].includes(id);
  const entries =
    hasId === enabled
      ? current[key]
      : enabled
        ? [...current[key], id]
        : current[key].filter((value) => value !== id);
  if (entries.length > MAX_SHELF_ENTRIES)
    throw new ShelfDataError(
      "This shelf is full. Export it before removing a few entries.",
    );
  const dismissed =
    key !== "dismissed" && enabled && current.dismissed.includes(id)
      ? current.dismissed.filter((value) => value !== id)
      : key === "dismissed"
        ? entries
        : current.dismissed;
  return { ...current, [key]: entries, dismissed };
}

type StorageReader = Pick<Storage, "getItem">;
export type StoredShelf = {
  preferences: ShelfPreferences;
  status: "ok" | "empty" | "invalid" | "unavailable";
};
export function readStoredShelf(storage: StorageReader | null): StoredShelf {
  if (!storage) return { preferences: emptyShelf(), status: "unavailable" };
  let raw: string | null;
  try {
    raw = storage.getItem(SHELF_STORAGE_KEY);
  } catch {
    return { preferences: emptyShelf(), status: "unavailable" };
  }
  if (raw === null) return { preferences: emptyShelf(), status: "empty" };
  try {
    return { preferences: validateShelf(parseJson(raw)), status: "ok" };
  } catch {
    return { preferences: emptyShelf(), status: "invalid" };
  }
}

export type ShelfSnapshot = {
  preferences: ShelfPreferences;
  ready: boolean;
  error: "invalid" | "unavailable" | "write" | null;
};
export const initialShelfSnapshot: ShelfSnapshot = {
  preferences: emptyShelf(),
  ready: false,
  error: null,
};
type ShelfOperation = (current: ShelfPreferences) => ShelfPreferences;
type ShelfStorage = Pick<Storage, "getItem" | "setItem">;

/** Serializes local actions and rebases pending session changes on the newest stored shelf. */
export function createShelfStore(options: {
  storage: () => ShelfStorage | null;
  lock?: (work: () => ShelfPreferences) => Promise<ShelfPreferences>;
}) {
  let snapshot = initialShelfSnapshot;
  let committed = emptyShelf();
  let pending: ShelfOperation[] = [];
  let queue = Promise.resolve();
  const listeners = new Set<() => void>();
  const emit = (
    preferences: ShelfPreferences,
    error: ShelfSnapshot["error"],
  ) => {
    const previous = snapshot.preferences;
    const shared = Object.fromEntries(
      SHELF_KEYS.map((key) => [
        key,
        previous[key].length === preferences[key].length &&
        previous[key].every((id, index) => id === preferences[key][index])
          ? previous[key]
          : preferences[key],
      ]),
    ) as ShelfPreferences;
    const unchanged = SHELF_KEYS.every((key) => shared[key] === previous[key]);
    if (unchanged && snapshot.ready && snapshot.error === error) return;
    snapshot = {
      preferences: unchanged ? previous : shared,
      ready: true,
      error,
    };
    listeners.forEach((listener) => listener());
  };
  const read = () => {
    let storage: ShelfStorage | null;
    try {
      storage = options.storage();
    } catch {
      storage = null;
    }
    const loaded = readStoredShelf(storage);
    if (loaded.status === "ok" || loaded.status === "empty")
      committed = loaded.preferences;
    return {
      storage,
      loaded,
      preferences: pending.reduce(
        (current, operation) => operation(current),
        committed,
      ),
    };
  };
  return {
    getSnapshot: () => snapshot,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    refresh: () => {
      try {
        const { loaded, preferences } = read();
        emit(
          preferences,
          loaded.status === "invalid"
            ? "invalid"
            : loaded.status === "unavailable"
              ? "unavailable"
              : pending.length
                ? "write"
                : null,
        );
      } catch {
        emit(snapshot.preferences, "write");
      }
    },
    update: (operation: ShelfOperation): Promise<ShelfPreferences> => {
      const work = () => {
        const { storage, loaded, preferences } = read();
        const next = operation(preferences);
        let error: ShelfSnapshot["error"] = null;
        if (loaded.status === "invalid")
          error = "invalid"; // Never overwrite an unreadable original.
        else if (!storage || loaded.status === "unavailable")
          error = "unavailable";
        else {
          try {
            storage.setItem(SHELF_STORAGE_KEY, JSON.stringify(next));
          } catch {
            error = "write";
          }
        }
        if (error) pending.push(operation);
        else {
          committed = next;
          pending = [];
        }
        emit(next, error);
        return next;
      };
      const result = queue.then(() =>
        options.lock ? options.lock(work) : work(),
      );
      // A rejected import must not prevent a later valid action from being processed.
      queue = result.then(
        () => undefined,
        () => undefined,
      );
      return result;
    },
  };
}
