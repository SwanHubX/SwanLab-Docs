import { defineConfig } from "vitepress";
import type { LocaleConfig } from "vitepress";

import type { Plugin, PluginOption } from "vite";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync } from "node:fs";
import path from "node:path";
import llmstxt from "vitepress-plugin-llms";
import { copyOrDownloadAsMarkdownButtons } from "vitepress-plugin-llms";
import { zh } from "./zh";
import { en } from "./en";

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

const plugins: PluginOption[] = [normalizeLegacyZhMarkdownRequests(), llmstxt()];
export default defineConfig({
  srcExclude: ["playground/**"],
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
