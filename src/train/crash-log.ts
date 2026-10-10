// Remembers which throughput-sweep configurations killed the tab.
//
// A worker pool can exhaust memory badly enough that the renderer dies — no
// exception, no unload event, nothing the page can catch. The only way to know
// is to write down what is about to be attempted BEFORE attempting it, and
// look for an entry with no matching completion on the next load.
//
// localStorage rather than OPFS on purpose: it is synchronous, so the record
// is durable before the risky work starts. An async write might not land.

const KEY = "wicroduck.sweep.attempts";

export interface Attempt {
  id: string;
  label: string;
  at: number;
  /** Set when the cell finished; absent means the tab died mid-cell. */
  done?: true;
}

function read(): Attempt[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Attempt[]) : [];
  } catch {
    return [];
  }
}

function write(list: Attempt[]): void {
  try {
    // Keep the tail: only recent history is interesting and the quota is small.
    localStorage.setItem(KEY, JSON.stringify(list.slice(-100)));
  } catch {
    /* private mode, quota — losing the log is not worth failing a sweep over */
  }
}

/** Configurations that were started and never finished. */
export function crashedIds(): Set<string> {
  return new Set(read().filter((a) => !a.done).map((a) => a.id));
}

export function crashedList(): Attempt[] {
  return read().filter((a) => !a.done);
}

export function markAttempt(id: string, label: string): void {
  write([...read().filter((a) => a.id !== id || a.done), { id, label, at: Date.now() }]);
}

export function markDone(id: string): void {
  write(read().map((a) => (a.id === id && !a.done ? { ...a, done: true as const } : a)));
}

export function clearCrashes(): void {
  write(read().filter((a) => a.done));
}
