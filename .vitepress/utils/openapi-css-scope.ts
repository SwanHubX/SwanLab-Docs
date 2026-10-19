import type { UserConfig } from "vite";

// Reuse Vite's own postcss types. Importing `Plugin` from "postcss" resolves
// to the npm-hoisted copy (8.5.26) instead of the pnpm copy Vite uses
// (8.5.29); the two are structurally unrelated, which makes tsc blow up
// while inferring the whole VitePress config ("excessively deep" TS2321).
type VitePostcss = Extract<NonNullable<UserConfig["css"]>["postcss"], object>;
type AcceptedPlugin = NonNullable<VitePostcss["plugins"]>[number];

// vitepress-openapi distributes a standalone Tailwind build. Its utilities are
// global and unlayered, so single utilities such as `.relative` / `.hidden` /
// `.gap-2` beat the theme's layered responsive + dark variants (e.g. the
// navbar's `lg:fixed`) and break the docs chrome.
//
// Scope those rules to the wrapper rendered by the admin API pages. Classes
// that belong to the vendor itself (`OASidebarItem` badges, `vjs-*` JSON
// viewer) stay global: they are unique, and the sidebar lives outside the
// page wrapper.
const SCOPE_CLASS = "swanlab-openapi";
const VENDOR_FILE = /[\\/]vitepress-openapi[\\/].*\.css$/;
const VENDOR_CLASS = /\.(?:OA[A-Z][\w-]*|oa-[\w-]+|vjs-[\w-]+)/;
const GLOBAL_SELECTOR = /^(?::root|\.dark|:host(?:\([^)]*\))?|::backdrop)$/;

function splitSelectorList(selector: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let quote: string | null = null;
  let start = 0;

  for (let i = 0; i < selector.length; i++) {
    const ch = selector[i];

    if (quote) {
      if (ch === "\\") i++;
      else if (ch === quote) quote = null;
      continue;
    }

    if (ch === '"' || ch === "'") quote = ch;
    else if (ch === "(" || ch === "[") depth++;
    else if (ch === ")" || ch === "]") depth--;
    else if (ch === "," && depth === 0) {
      parts.push(selector.slice(start, i));
      start = i + 1;
    }
  }

  parts.push(selector.slice(start));
  return parts;
}

export function scopeOpenapiCss(): AcceptedPlugin {
  return {
    postcssPlugin: "swanlab-scope-openapi-css",
    Once(root) {
      root.walkRules((rule) => {
        const file = rule.source?.input?.file;
        if (!file || !VENDOR_FILE.test(file)) return;

        const parent = rule.parent;
        if (parent?.type === "atrule" && parent.name.includes("keyframes")) return;

        const selectors = splitSelectorList(rule.selector);
        if (selectors.some((sel) => VENDOR_CLASS.test(sel))) return;
        if (selectors.some((sel) => GLOBAL_SELECTOR.test(sel.trim()))) return;

        rule.selector = selectors.map((sel) => `.${SCOPE_CLASS} ${sel.trim()}`).join(", ");
      });
    },
  };
}
