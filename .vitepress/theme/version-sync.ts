/**
 * Browser-side version sync for the navbar version dropdown.
 *
 * The build bakes `SWANLAB_VERSION` into the nav (see .vitepress/version.ts),
 * so the first paint always has a version and never flashes. This layer keeps
 * long-lived deployments fresh without rebuilds:
 *
 *  - localStorage cache `{ version, fetchedAt }` with a 6h TTL.
 *  - Within the TTL the cached version is applied straight away (when newer
 *    than the baked one) — no network request at all.
 *  - Once the TTL expires, one silent background fetch to PyPI refreshes the
 *    cache; if a newer version comes back the navbar is patched in place.
 *    Each browser therefore hits PyPI at most once per 6h, and only when
 *    someone actually visits.
 *
 * Patching is monotonic (never downgrades the displayed version) and
 * incremental: it mutates the nav theme config so future re-renders (route
 * changes, mobile menu) keep the new version, and patches the already-rendered
 * DOM so the update shows immediately. Everything is best-effort — storage or
 * network failures are swallowed and the baked version simply keeps showing.
 */
import { useData } from "vitepress";

const PYPI_URL = "https://pypi.org/pypi/swanlab/json";
const STORAGE_KEY = "swanlab:version";
const TTL_MS = 6 * 60 * 60 * 1000; // 6h
const FETCH_TIMEOUT_MS = 5000;

type VersionCache = { version: string; fetchedAt: number };

/** Version currently shown in the navbar; only ever moves forward. */
let currentVersion = __SWANLAB_VERSION__;

function readCache(): VersionCache | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed.version === "string" && typeof parsed.fetchedAt === "number") {
      return parsed as VersionCache;
    }
    return null;
  } catch {
    return null;
  }
}

function writeCache(version: string): void {
  try {
    const entry: VersionCache = { version, fetchedAt: Date.now() };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entry));
  } catch {
    /* caching is best-effort; ignore write failures */
  }
}

/** Dotted-numeric compare; true when `next` is strictly newer than `current`. */
function isNewer(next: string, current: string): boolean {
  const parse = (v: string) => v.split(/[.-]/).map((s) => parseInt(s, 10) || 0);
  const a = parse(next);
  const b = parse(current);
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    if (x !== y) return x > y;
  }
  return false;
}

async function fetchLatestVersion(): Promise<string | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    const res = await fetch(PYPI_URL, { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) return null;
    const data = (await res.json()) as { info?: { version?: string } };
    const v = data?.info?.version;
    return typeof v === "string" && v.length > 0 ? v : null;
  } catch {
    return null;
  }
}

/**
 * Wires the version sync. Must be called in the theme's `setup()` (uses
 * `useData`); returns the init function to invoke on `onMounted`.
 */
export function useVersionSync(): () => void {
  const { themeConfig } = useData();

  /** Patch nav config + live DOM. Idempotent and never downgrades. */
  function applyVersion(version: string): void {
    if (!isNewer(version, currentVersion)) return;
    const oldText = `v${currentVersion}`;
    const newText = `v${version}`;
    currentVersion = version;

    // 1) Mutate the nav config so any future re-render (route change, mobile
    //    nav screen open) picks up the new version.
    const nav = themeConfig.value.nav as Array<{ text?: string }> | undefined;
    const item = nav?.find((i) => typeof i.text === "string" && /^v\d/.test(i.text));
    if (item) item.text = newText;

    // 2) Patch the already-rendered DOM immediately — the config above is not
    //    reactive, and waiting for the next route change would delay the update.
    //    Scoped to the navbar / nav screen so page content is never touched.
    for (const root of document.querySelectorAll("header.VPNav, .VPNavScreen")) {
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      let node: Node | null;
      while ((node = walker.nextNode())) {
        if (node.nodeValue?.trim() === oldText) {
          node.nodeValue = newText;
        }
      }
    }
  }

  /** Silent background refresh; writes the cache even when not newer, so the TTL resets. */
  async function refresh(): Promise<void> {
    const latest = await fetchLatestVersion();
    if (!latest) return;
    writeCache(latest);
    applyVersion(latest);
  }

  return function initVersionSync(): void {
    const cached = readCache();
    // TTL 内直接用缓存覆盖（不比烤入版本新则 applyVersion 内部忽略）。
    if (cached) applyVersion(cached.version);
    // 缓存缺失或过期才静默 fetch，保证每个浏览器最多每 6h 一次请求。
    if (!cached || Date.now() - cached.fetchedAt >= TTL_MS) {
      void refresh();
    }
  };
}
