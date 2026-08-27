// Post-build enhancement for generated HTML heads. Runs after `vitepress build`
// (see package.json docs:build) because VitePress emits a single site-wide
// <meta name="description"> that can only be replaced, not appended to via
// transformHead, and VitePress 2.0-alpha serializes config functions with
// toString() making hook-side file access fragile.
//
// Per page it:
//   1. Replaces the description meta with text extracted from the page's own
//      markdown intro (falls back to the locale default).
//   2. Injects Open Graph / Twitter Card tags so shared links render cards.
//
// The script is idempotent: previously injected blocks (between markers) are
// stripped before regenerating.
import { readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

const SITE_URL = "https://docs.swanlab.cn";
const OG_IMAGE = `${SITE_URL}/page.png`;

// Keep in sync with the description fields in .vitepress/zh.ts / en.ts.
const LOCALE_DEFAULTS = {
  zh: {
    description: "SwanLab官方文档, 提供最全面的使用指南和API文档",
    siteName: "SwanLab官方文档",
    ogLocale: "zh_CN",
  },
  en: {
    description:
      "SwanLab Official Documentation, providing the most comprehensive user guide and API documentation",
    siteName: "SwanLab Docs",
    ogLocale: "en_US",
  },
};

function walkHtmlFiles(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stat = statSync(full);
    if (stat.isDirectory()) {
      walkHtmlFiles(full, out);
    } else if (entry.endsWith(".html")) {
      out.push(full);
    }
  }
  return out;
}

function stripBlockLevelMarkdown(markdown) {
  return markdown
    .replace(/^---\r?\n[\s\S]*?\r?\n---/, "") // frontmatter
    .replace(/```[\s\S]*?(?:```|$)/g, "\n") // fenced code
    .replace(/~~~[\s\S]*?(?:~~~|$)/g, "\n")
    .replace(/<script\b[\s\S]*?<\/script>/gi, "\n")
    .replace(/<style\b[\s\S]*?<\/style>/gi, "\n")
    .replace(/<!--[\s\S]*?-->/g, "\n") // html comments / Vue template comments
    .replace(/^\s*import\s.*$/gm, ""); // md imports of Vue components etc.
}

function inlineToText(line) {
  return line
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "") // images: drop entirely
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1") // links -> label
    .replace(/`{1,3}([^`]*)`{1,3}/g, "$1") // inline code -> content
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/_{1,2}([^_]+)_{1,2}/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

// Extract an ~150 char summary from the document: the first prose-looking
// lines that survive the structural filters (code fences, tables, lists,
// headings and component tags are all stripped first).
function extractDescription(markdown) {
  const cleaned = stripBlockLevelMarkdown(markdown);
  let collected = "";

  for (const rawLine of cleaned.split("\n")) {
    const line = rawLine.trim();

    if (/^#{1,6}\s/.test(line) || !line) {
      continue;
    }
    // Structural lines that read badly as summaries: tables, quotes, lists,
    // frontmatter-ish directives, Vue components/JSX. API reference pages put
    // their prose *after* the signature code block and parameter table, so we
    // deliberately do not bound the search to the top of the document — the
    // structural filters above already keep tables/code out.
    if (/^[|>{\-[*[<]/.test(line) || /^\d+[.)]\s/.test(line)) {
      continue;
    }

    const text = inlineToText(line);
    // Below this length the line is usually a badge caption or stray fragment.
    if (text.length < 16) {
      continue;
    }
    collected += (collected ? "" : " ") + text;
    if (collected.length >= 90) {
      break;
    }
  }

  collected = collected.trim().slice(0, 156);
  return collected.length >= 16 ? collected : null;
}

function resolveSourceMarkdown(relPath) {
  // Mirror VitePress rewrites: html paths under en/ map back to en/<route>.md,
  // everything else to zh/<route>.md. Prefer the page's own locale so en pages
  // never inherit descriptions translated out of zh content; the other locale
  // only serves as fallback for pages missing on that side (i18nRouting is off,
  // so untranslated routes already display the other language).
  const route = relPath.replace(/\.html$/, ".md").replace(/^en\//, "");
  const candidates = relPath.startsWith("en/")
    ? [`en/${route}`, `zh/${route}`]
    : [`zh/${route}`, `en/${route}`];
  for (const candidate of candidates) {
    try {
      return readFileSync(candidate, "utf8");
    } catch {
      // fall through to next candidate
    }
  }
  return null;
}

function unescapeHtml(text) {
  return text
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .replaceAll("&apos;", "'");
}

function escapeHtmlAttr(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function canonicalUrl(relPath) {
  const urlPath = relPath.replace(/(^|\/)index\.html$/, "$1");
  return `${SITE_URL}/${urlPath}`;
}

const DESCRIPTION_META_RE = /<meta\s+name="description"\s+content="[^"]*"\s*>/i;
const TITLE_RE = /<title>([\s\S]*?)<\/title>/i;
const INJECTED_BLOCK_RE =
  /\n?<!-- swanlab-head-enhance -->\n[\s\S]*?<!-- \/swanlab-head-enhance -->\n?/;

let extracted = 0;
let defaulted = 0;

function enhancePage(distDir, absPath) {
  const relPath = path.relative(distDir, absPath).split(path.sep).join("/");
  const locale = relPath.startsWith("en/") ? LOCALE_DEFAULTS.en : LOCALE_DEFAULTS.zh;
  let html = readFileSync(absPath, "utf8");

  // Idempotency: drop any block injected by a previous run first.
  html = html.replace(INJECTED_BLOCK_RE, "\n");

  const sourceMd = resolveSourceMarkdown(relPath);
  const autoDescription = sourceMd && extractDescription(sourceMd);
  if (autoDescription) {
    extracted += 1;
  } else {
    defaulted += 1;
    if (process.env.SWANLAB_SEO_DEBUG) {
      console.log(`[enhance-doc-head] no description: ${relPath}`);
    }
  }
  const description = escapeHtmlAttr(autoDescription ?? locale.description);

  const titleMatch = html.match(TITLE_RE);
  const ogTitle = escapeHtmlAttr(unescapeHtml(titleMatch?.[1]?.trim() || locale.siteName));

  const seoBlock = [
    "<!-- swanlab-head-enhance -->",
    `<meta name="description" content="${description}">`,
    '<meta property="og:type" content="website">',
    `<meta property="og:site_name" content="${escapeHtmlAttr(locale.siteName)}">`,
    `<meta property="og:title" content="${ogTitle}">`,
    `<meta property="og:description" content="${description}">`,
    `<meta property="og:url" content="${canonicalUrl(relPath)}">`,
    `<meta property="og:image" content="${OG_IMAGE}">`,
    `<meta property="og:locale" content="${locale.ogLocale}">`,
    '<meta name="twitter:card" content="summary_large_image">',
    `<meta name="twitter:title" content="${ogTitle}">`,
    `<meta name="twitter:description" content="${description}">`,
    `<meta name="twitter:image" content="${OG_IMAGE}">`,
    "<!-- /swanlab-head-enhance -->",
  ].join("\n");

  if (DESCRIPTION_META_RE.test(html)) {
    html = html.replace(DESCRIPTION_META_RE, seoBlock);
  } else {
    html = html.replace("</head>", `${seoBlock}\n</head>`);
  }

  writeFileSync(absPath, html);
  return true;
}

const distDir = process.argv[2] ?? ".vitepress/dist";
const pages = walkHtmlFiles(distDir);
for (const page of pages) {
  enhancePage(distDir, page);
}
console.log(
  `[enhance-doc-head] ${pages.length} pages: ${extracted} descriptions extracted from markdown, ${defaulted} using locale default.`,
);
