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
 * Patching is monotonic (never downgrades the displayed version). Because
 * VitePress wraps site/theme data in `readonly()` (vitepress data.ts), we
 * cannot mutate the nav theme config to make the dropdown trigger reactive.
 * Instead we patch the rendered text nodes directly and keep a MutationObserver
 * on the navbar so the patch survives Vue re-renders (route changes, locale
 * switches, mobile menu toggle) — those always restore the baked version.
 * Everything is best-effort — storage or network failures are swallowed and
 * the baked version simply keeps showing.
 */
const PYPI_URL = "https://pypi.org/pypi/swanlab/json";
const STORAGE_KEY = "swanlab:version";
const TTL_MS = 6 * 60 * 60 * 1000; // 6h
const FETCH_TIMEOUT_MS = 5000;

/** Build-time version baked into the HTML; the baseline text nodes always carry this. */
const BAKED_VERSION = __SWANLAB_VERSION__;

type VersionCache = { version: string; fetchedAt: number };

/** Version to display in the navbar; only ever moves forward. */
let displayVersion = BAKED_VERSION;

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
 * Patch the baked version text in the rendered navbar DOM. Idempotent —
 * finds every text node matching the baked version and replaces it with the
 * display version. Scoped to the navbar / nav screen so page content is
 * never touched.
 */
function patchDom(): void {
  const bakedText = `v${BAKED_VERSION}`;
  const newText = `v${displayVersion}`;
  if (bakedText === newText) return;

  for (const root of document.querySelectorAll("header.VPNav, .VPNavScreen")) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node: Node | null;
    while ((node = walker.nextNode())) {
      if (node.nodeValue?.trim() === bakedText) {
        node.nodeValue = newText;
      }
    }
  }
}

/** Set the display version (monotonic) and patch the DOM immediately. */
function applyVersion(version: string): void {
  if (!isNewer(version, displayVersion)) return;
  displayVersion = version;
  patchDom();
}

/** Silent background refresh; writes the cache even when not newer, so the TTL resets. */
async function refresh(): Promise<void> {
  const latest = await fetchLatestVersion();
  if (!latest) return;
  writeCache(latest);
  applyVersion(latest);
}

/**
 * Returns the init function to invoke on `onMounted`. No longer needs
 * `useData()` — the theme config is readonly and patched via DOM instead.
 */
export function useVersionSync(): () => void {
  return function initVersionSync(): void {
    const cached = readCache();
    // TTL 内直接用缓存覆盖（不比烤入版本新则 applyVersion 内部忽略）。
    if (cached) applyVersion(cached.version);
    // 缓存缺失或过期才静默 fetch，保证每个浏览器最多每 6h 一次请求。
    if (!cached || Date.now() - cached.fetchedAt >= TTL_MS) {
      void refresh();
    }

    // VitePress exposes theme config as readonly(), so Vue re-renders (route
    // changes, locale switches, mobile menu) always restore the baked version.
    // Watch the navbar and re-patch whenever that happens.
    const nav = document.querySelector("header.VPNav");
    if (nav) {
      new MutationObserver(() => patchDom()).observe(nav, {
        subtree: true,
        childList: true,
        characterData: true,
      });
    }
  };
}
