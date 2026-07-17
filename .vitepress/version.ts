/**
 * Build-time SwanLab version resolver.
 *
 * Fetches the latest published version from PyPI once per build/dev-start and
 * exposes it as `SWANLAB_VERSION`, so the navbar version dropdown (`zh.ts` /
 * `en.ts`) tracks releases automatically — no manual edit per release.
 *
 * Frequency control (the request must NOT fire on every page visit):
 *  - The result is cached in `.vitepress/cache/swanlab-version.json`
 *    (gitignored). Rebuilds / dev-server restarts within the TTL reuse the
 *    cache instead of hitting PyPI again.
 *  - There is NO client-side fetch — the version is baked into the static
 *    build, so end users never trigger a PyPI request. Freshness on the
 *    deployed site is therefore tied to how often docs are rebuilt/deployed
 *    (point CI/deploy at this repo and it stays current).
 *
 * Robustness:
 *  - 3s request timeout; on any failure we fall back to the stale cache, or to
 *    FALLBACK_VERSION if no cache exists yet — the build never blocks on PyPI.
 *
 * Resolved via top-level await so the value is a plain string by the time
 * zh.ts / en.ts build the nav config (no runtime cost downstream).
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import path from "node:path";
import https from "node:https";

const PYPI_URL = "https://pypi.org/pypi/swanlab/json";
// process.cwd() is the VitePress docs root (the dir holding .vitepress/) and
// is stable regardless of how the config gets bundled — unlike import.meta.url,
// which would point at a temp bundle file after esbuild packs the config.
const CACHE_FILE = path.resolve(process.cwd(), ".vitepress", "cache", "swanlab-version.json");
// Used only when there is no cache yet AND PyPI is unreachable.
const FALLBACK_VERSION = "0.9.0";
const TTL_MS = 30 * 60 * 1000; // 30 min
const FETCH_TIMEOUT_MS = 3000;

type VersionCache = { version: string; fetchedAt: number };

function readCache(): VersionCache | null {
  try {
    if (!existsSync(CACHE_FILE)) return null;
    const parsed = JSON.parse(readFileSync(CACHE_FILE, "utf8"));
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
    mkdirSync(path.dirname(CACHE_FILE), { recursive: true });
    writeFileSync(
      CACHE_FILE,
      JSON.stringify({
        version,
        fetchedAt: Date.now(),
      } as VersionCache),
    );
  } catch {
    /* caching is best-effort; ignore write failures */
  }
}

function fetchLatestVersion(): Promise<string | null> {
  return new Promise((resolve) => {
    const req = https.get(PYPI_URL, { timeout: FETCH_TIMEOUT_MS }, (res) => {
      if (res.statusCode !== 200) {
        res.resume();
        resolve(null);
        return;
      }
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk: string) => {
        body += chunk;
      });
      res.on("end", () => {
        try {
          const data = JSON.parse(body) as {
            info?: { version?: string };
          };
          const v = data?.info?.version;
          resolve(typeof v === "string" && v.length > 0 ? v : null);
        } catch {
          resolve(null);
        }
      });
    });
    req.on("error", () => resolve(null));
    req.on("timeout", () => {
      req.destroy();
      resolve(null);
    });
  });
}

async function resolveVersion(): Promise<string> {
  const cached = readCache();
  if (cached && Date.now() - cached.fetchedAt < TTL_MS) {
    return cached.version;
  }

  const latest = await fetchLatestVersion();
  if (latest) {
    writeCache(latest);
    return latest;
  }

  // PyPI unreachable — prefer a stale cache over the hardcoded fallback so a
  // transient outage never rolls the displayed version backwards.
  return cached?.version ?? FALLBACK_VERSION;
}

/** Latest SwanLab version, resolved once when the config is loaded. */
export const SWANLAB_VERSION = await resolveVersion();
