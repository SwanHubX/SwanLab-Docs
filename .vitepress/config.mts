import { defineConfig } from "vitepress";
import type { LocaleConfig } from "vitepress";

import type { Plugin, PluginOption } from "vite";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import llmstxt from "vitepress-plugin-llms";
import { copyOrDownloadAsMarkdownButtons } from "vitepress-plugin-llms";
import { groupIconMdPlugin, groupIconVitePlugin } from "vitepress-plugin-group-icons";
import { zh } from "./zh";
import { en } from "./en";

const srcExclude = ["playground/**", "AGENTS.md", "README.md", "TRICK.md"];

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
  llmstxt(),
  // Code-block tab icons (pip/conda/python/bash etc.) — see vitepress-plugin-group-icons
  groupIconVitePlugin({}),
];
export default defineConfig({
  srcExclude,
  cleanUrls: true,
  sitemap: {
    hostname: "https://docs.swanlab.cn",
  },
  vite: { plugins },

  rewrites(id) {
    return id.startsWith("zh/") ? id.slice(3) : id;
  },

  themeConfig: {
    search: {
      provider: "local",
    },
  },

  markdown: {
    config(md) {
      md.use(copyOrDownloadAsMarkdownButtons);
      // Tab icons on grouped/single code blocks
      md.use(groupIconMdPlugin, { titleBar: { includeSnippet: true } });
    },
    image: {
      lazyLoading: true,
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
});
