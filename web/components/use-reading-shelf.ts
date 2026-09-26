"use client";

import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import {
  createShelfStore,
  initialShelfSnapshot,
  mergeShelves,
  setShelfPreference,
  SHELF_STORAGE_KEY,
  type ShelfKey,
  type ShelfPreferences,
} from "@/lib/shelf";

function browserStorage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function useReadingShelf() {
  const [store] = useState(() =>
    createShelfStore({
      storage: browserStorage,
      lock: async (work) =>
        navigator.locks
          ? navigator.locks.request(SHELF_STORAGE_KEY, work)
          : work(),
    }),
  );
  const snapshot = useSyncExternalStore(
    store.subscribe,
    store.getSnapshot,
    () => initialShelfSnapshot,
  );
  useEffect(() => {
    store.refresh();
    const onStorage = (event: StorageEvent) => {
      if (
        (event.key === SHELF_STORAGE_KEY || event.key === null) &&
        (!event.storageArea || event.storageArea === browserStorage())
      )
        store.refresh();
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [store]);
  const toggle = useCallback(
    async (key: ShelfKey, id: string) => {
      // Intent is captured when clicked, then rebased on the latest shelf inside the lock.
      const enabled = !store.getSnapshot().preferences[key].includes(id);
      await store.update((current) =>
        setShelfPreference(current, key, id, enabled),
      );
      return enabled;
    },
    [store],
  );
  const importPreferences = useCallback(
    async (incoming: ShelfPreferences) => {
      let report: ReturnType<typeof mergeShelves> | undefined;
      await store.update((current) => {
        report = mergeShelves(current, incoming);
        return report.preferences;
      });
      return report!;
    },
    [store],
  );
  return { ...snapshot, toggle, importPreferences };
}
