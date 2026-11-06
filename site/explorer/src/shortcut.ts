// Cmd+K on a Mac, Ctrl+K elsewhere, focuses the search, as in the Network
// Navigator and on documentation sites.
const MAC =
  typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad/.test(navigator.platform);

/** The search shortcut as its key cap reads. */
export const SEARCH_KEYS = MAC ? "⌘K" : "Ctrl K";

/** The same shortcut in `aria-keyshortcuts` notation. */
export const SEARCH_ARIA = MAC ? "Meta+K" : "Control+K";

export function isSearchShortcut(event: KeyboardEvent): boolean {
  return (
    event.key.toLowerCase() === "k" &&
    (MAC ? event.metaKey : event.ctrlKey) &&
    !event.altKey &&
    !event.shiftKey
  );
}
