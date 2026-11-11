import { defineConfig } from "vitepress";
import type { LocaleConfig } from "vitepress";

import type { Plugin, PluginOption } from "vite";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import llmstxt from "vitepress-plugin-llms";
import { copyOrDownloadAsMarkdownButtons } from "vitepress-plugin-llms";
import { groupIconMdPlugin, groupIconVitePlugin } from "vitepress-plugin-group-icons";
import { extendConfig } from "@voidzero-dev/vitepress-theme/config";
import { scopeOpenapiCss } from "./utils/openapi-css-scope.ts";
import { zh } from "./zh.ts";
import { en } from "./en.ts";
import { SWANLAB_VERSION } from "./version.ts";

const srcExclude = ["playground/**", "AGENTS.md", "CLAUDE.md", "README.md", "TRICK.md"];

function normalizePath(filePath: string) {
  return filePath.split(path.sep).join("/");
}

function listMarkdownFiles(directory: string): string[] {
  const root = path.resolve(process.cwd(), directory);
  const files: string[] = [];

  function walk(current: string) {
    let entries;

    try {
      entries = readdirSync(current, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);

      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile() && entry.name.endsWith(".md")) {
        files.push(normalizePath(path.relative(process.cwd(), fullPath)));
      }
    }
  }

  if (statSync(root, { throwIfNoEntry: false })?.isDirectory()) {
    walk(root);
  }

  return files;
}

function markdownPathToRoute(filePath: string) {
  const rewrittenPath = filePath.startsWith("zh/") ? filePath.slice(3) : filePath;
  const route = rewrittenPath.replace(/(^|\/)index\.md$/, "$1").replace(/\.md$/, "");

  return route === "" ? "/" : route;
}

const localizedRoutes = {
  root: new Set(listMarkdownFiles("zh").map(markdownPathToRoute)),
  en: new Set(listMarkdownFiles("en").map(markdownPathToRoute)),
};

const rootOnlyRoutes = new Set(
  Array.from(localizedRoutes.root).filter(
    (route) => route !== "/" && !localizedRoutes.en.has(`en/${route}`),
  ),
);

type AdditionalConfigLoader = (relativePath: string) =>
  | [
      {
        themeConfig: {
          i18nRouting: false;
        };
      },
    ]
  | undefined;

function createRootOnlyAdditionalConfig(routes: Set<string>): AdditionalConfigLoader {
  const serializedRoutes = JSON.stringify(Array.from(routes).sort());

  // VitePress serializes config functions with toString(), so the route list
  // must be embedded in the function body instead of captured from this file.
  // oxlint-disable-next-line typescript/no-implied-eval
  return new Function(`
    return function additionalConfig(relativePath) {
      const rootOnlyRoutes = new Set(${serializedRoutes});
      const rewrittenPath = relativePath.startsWith("zh/") ? relativePath.slice(3) : relativePath;
      const route = rewrittenPath.replace(/(^|\\/)index\\.md$/, "$1").replace(/\\.md$/, "");
      const normalizedRoute = route === "" ? "/" : route;

      if (rootOnlyRoutes.has(normalizedRoute)) {
        return [
          {
            themeConfig: {
              i18nRouting: false,
            },
          },
        ];
      }
    };
  `)() as AdditionalConfigLoader;
}

const rootOnlyAdditionalConfig = createRootOnlyAdditionalConfig(rootOnlyRoutes);

function readTextFileInside(root: string, requestPath: string) {
  const resolvedRoot = path.resolve(root);
  const resolvedFile = path.resolve(resolvedRoot, requestPath.replace(/^\/+/, ""));

  if (resolvedFile !== resolvedRoot && !resolvedFile.startsWith(resolvedRoot + path.sep)) {
    return;
  }

  try {
    return readFileSync(resolvedFile, "utf8");
  } catch {
    return;
  }
}

function normalizeLegacyZhMarkdownRequests(): Plugin {
  return {
    name: "swanlab:normalize-legacy-zh-markdown-requests",
    enforce: "pre",
    configureServer(server) {
      const normalizeMarkdownRequest = (
        req: IncomingMessage,
        res: ServerResponse,
        next: (err?: unknown) => void,
      ) => {
        const url = req.url;

        if (url) {
          const queryIndex = url.indexOf("?");
          const pathname = queryIndex === -1 ? url : url.slice(0, queryIndex);

          if (queryIndex === -1 && /\.(?:md|txt)$/.test(pathname)) {
            const text =
              readTextFileInside(process.cwd(), pathname) ??
              readTextFileInside(process.cwd(), "/zh" + pathname) ??
              readTextFileInside(path.resolve(process.cwd(), ".vitepress/dist"), pathname);

            if (text !== undefined) {
              res.setHeader("Content-Type", "text/plain; charset=utf-8");
              res.end(text);
              return;
            }
          }
        }

        next();
      };

      server.middlewares.use(normalizeMarkdownRequest);

      const stack = (server.middlewares as typeof server.middlewares & { stack?: unknown[] }).stack;
      const layer = stack?.pop();

      if (layer) {
        stack?.unshift(layer);
      }
    },
  };
}

const plugins: PluginOption[] = [
  normalizeLegacyZhMarkdownRequests(),
  llmstxt({
    generateLLMsFullTxt: false,
  }),
  // Code-block tab icons — see vitepress-plugin-group-icons. The builtin
  // vscode-icons logos (npm/bun/...) render noisy at 14px tab size; swap the
  // package-manager family (plus python) for skill-icons' badge versions,
  // which ship dedicated light/dark variants (@iconify-json/skill-icons).
  // js/golang/curl serve the REST API code-sample tabs (see theme/index.ts
  // availableLanguages); defaultLabels guarantees their CSS is generated —
  // those tabs live on dynamic routes the plugin's markdown scan can't see.
  groupIconVitePlugin({
    customIcon: {
      npm: { light: "skill-icons:npm-light", dark: "skill-icons:npm-dark" },
      bun: { light: "skill-icons:bun-light", dark: "skill-icons:bun-dark" },
      pnpm: { light: "skill-icons:pnpm-light", dark: "skill-icons:pnpm-dark" },
      yarn: { light: "skill-icons:yarn-light", dark: "skill-icons:yarn-dark" },
      deno: { light: "skill-icons:deno-light", dark: "skill-icons:deno-dark" },
      python: { light: "skill-icons:python-light", dark: "skill-icons:python-dark" },
      js: "skill-icons:javascript",
      golang: "skill-icons:golang",
      // cURL tab：skill-icons 没有 curl 图标，logos:curl 是 512×154 的深色
      // 宽字标——1em 方框里过小且暗色模式不可见；用 bash 徽章（shell 语义）
      // 保持与其他 tab 一致的方徽章家族和官方深浅变体。
      curl: { light: "skill-icons:bash-light", dark: "skill-icons:bash-dark" },
    },
    defaultLabels: ["curl", "js", "python", "golang"],
  }),
];
// Wrapped with the VoidZero theme's extendConfig, which injects the
// Tailwind plugin, @vp-default aliases, and asset handling.
export default extendConfig(
  defineConfig({
    srcExclude,
    cleanUrls: false,
    sitemap: {
      hostname: "https://docs.swanlab.cn",
    },
    vite: {
      plugins,
      optimizeDeps: {
        // VitePress hands Vite no scan entries, so anything outside its own
        // include list is only discovered on the first browser request: Vite
        // then optimizes it and forces a full page reload ("optimized
        // dependencies changed. reloading") on every cold start.
        include: [
          // Dynamically imported by the theme
          "medium-zoom",
          "mermaid",
          "vitepress-openapi/client",
          // extendConfig() excludes @voidzero-dev/vitepress-theme from
          // optimization, so its @vp-* aliases (TS sources inside the package)
          // get optimized one by one at runtime instead of up front.
          "@vp-composables/data",
          "@vp-composables/edit-link",
          "@vp-composables/flyout",
          "@vp-composables/icon",
          "@vp-composables/langs",
          "@vp-composables/layout",
          "@vp-composables/outline",
          "@vp-composables/prev-next",
          "@vp-composables/sidebar",
          "@vp-composables/sponsor-grid",
          "@vp-support/docsearch",
          "@vp-support/reactivity",
          "@vp-support/search-config",
          "@vp-support/utils",
        ],
      },
      css: {
        postcss: {
          // Keep vitepress-openapi's standalone Tailwind build from leaking
          // onto the docs chrome (it breaks the sticky navbar and dark mode
          // logo variants). See openapi-css-scope.ts.
          plugins: [scopeOpenapiCss()],
        },
      },
      define: {
        // Baked-in build-time version, read by the browser-side sync layer
        // (theme/version-sync.ts) as the baseline it may incrementally patch.
        __SWANLAB_VERSION__: JSON.stringify(SWANLAB_VERSION),
      },
    },

    rewrites(id) {
      return id.startsWith("zh/") ? id.slice(3) : id;
    },

    transformPageData(pageData) {
      // vitepress-openapi：动态路由生成的端点页通过 params 传递标题
      if (pageData.params?.pageTitle) {
        pageData.title = pageData.params.pageTitle;
      }
    },

    themeConfig: {
      // VoidZero theme variant — neutral branding for now (brand colors are
      // intentionally left untuned; see .vitepress/theme/styles.css).
      variant: "voidzero",
      search: {
        provider: "local",
      },
    },

    markdown: {
      // Code syntax theme — One Light / One Dark Pro. The block chrome colors
      // (background / dividers / copy pill) are aligned to this pair in
      // theme/styles/tokens.css.
      theme: { light: "one-light", dark: "one-dark-pro" },
      config(md) {
        // VitePress 2.0.0-alpha.19 caches the markdown-it instance in a
        // module-level singleton, and the content plugin and local-search
        // plugin race to create it during configResolved — both can end up
        // running this config on the surviving instance. Keep customizations
        // idempotent per instance.
        if ((md as any).__swanlabConfigured) return;
        (md as any).__swanlabConfigured = true;

        md.use(copyOrDownloadAsMarkdownButtons);
        // Tab icons on grouped/single code blocks
        md.use(groupIconMdPlugin, { titleBar: { includeSnippet: true } });

        // Render Mermaid fences as a client-side component. Mermaid itself is
        // loaded lazily, so pages without diagrams do not download it.
        const defaultFence = md.renderer.rules.fence!;
        md.renderer.rules.fence = (tokens, index, options, env, self) => {
          const token = tokens[index];

          if (token.info.trim() === "mermaid") {
            return `<MermaidDiagram graph="${encodeURIComponent(token.content)}" />`;
          }

          return defaultFence(tokens, index, options, env, self);
        };
      },
      image: {
        lazyLoad: true,
      },
      math: true,
    },

    locales: {
      root: { label: "简体中文", ...(zh as LocaleConfig) },
      en: { label: "English", ...(en as LocaleConfig) },
    },

    additionalConfig: rootOnlyAdditionalConfig,

    head: [
      [
        "script",
        {
          defer: "",
          src: "https://umami.swanlab.cn/script.js",
          "data-website-id": process.env.UMAMI_WEBSITE_ID ?? "",
        },
      ],
    ],
  }),
);
