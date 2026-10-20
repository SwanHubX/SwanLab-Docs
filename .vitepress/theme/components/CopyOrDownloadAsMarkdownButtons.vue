<template>
  <div class="markdown-copy-buttons">
    <div class="markdown-copy-buttons-inner">
      <AskAiDropdown :items="menuItems" />
      <button type="button" class="copy-page" @click="copyAsMarkdown">
        <span v-html="icons[copied ? 'check' : 'copy']" class="icon"></span>
        <span class="label">
          {{ copied ? "Copied" : "Copy Markdown" }}
        </span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { onUnmounted, ref } from "vue";
import AskAiDropdown from "./AskAiDropdown.vue";

const rawModules = import.meta.glob("../icons/*.svg", {
  eager: true,
  query: "?raw",
  import: "default",
});

const icons = Object.fromEntries(
  Object.entries(rawModules).map(([path, content]) => {
    const name = path.match(/\/([^/]+)\.svg$/)[1];
    return [name, content];
  }),
);

const aiProviders = [
  {
    icon: "chatgpt",
    name: "ChatGPT",
    promptUrl: "https://chatgpt.com/?hints=search&prompt=",
  },
  {
    icon: "claude",
    name: "Claude",
    promptUrl: "https://claude.ai/new?q=",
  },
  {
    icon: "kimi",
    name: "Kimi",
    promptUrl: "https://www.kimi.com/_prefill_chat?prefill_prompt=",
  },
];

function buildPrompt(sourceUrl) {
  return `Read from ${sourceUrl} so I can ask questions about it.`;
}

const menuItems = [
  { icon: "markdown", label: "View as Markdown", onSelect: viewAsMarkdown },
  ...aiProviders.map((provider) => ({
    icon: provider.icon,
    label: `Open in ${provider.name}`,
    onSelect: () => openInAI(provider),
  })),
];

const copied = ref(false);

let copiedTimer;

/** 「已复制」状态的停留时长。 */
const COPIED_RESET_DELAY = 2000;

function removeHtmlExtension(pathSegment) {
  const lastSlashIndex = pathSegment.lastIndexOf("/");
  const lastDotIndex = pathSegment.lastIndexOf(".");

  if (lastDotIndex > lastSlashIndex && lastDotIndex !== -1 && pathSegment.endsWith(".html")) {
    return pathSegment.slice(0, lastDotIndex);
  }

  return pathSegment;
}

function cleanUrl(url) {
  const { origin, pathname } = new URL(url);
  const pathnameWithoutTrailingSlash = pathname.replace(/\/+$/, "");

  if (pathname.length > 0) {
    return origin + removeHtmlExtension(pathnameWithoutTrailingSlash);
  }

  return origin;
}

function resolveMarkdownPageURL(url) {
  const cleanedURL = cleanUrl(url);

  if (cleanedURL === window.location.origin) {
    return `${cleanedURL}/index.md`;
  }

  return `${cleanedURL}.md`;
}

function normalizeLegacyZhPathname(pathname) {
  if (pathname === "/zh") {
    return "/";
  }

  if (pathname.startsWith("/zh/")) {
    return pathname.slice(3);
  }

  return pathname;
}

function getMarkdownURL() {
  const currentURL = window.location.origin + normalizeLegacyZhPathname(window.location.pathname);

  return resolveMarkdownPageURL(currentURL);
}

async function fetchMarkdownText(markdownUrl) {
  const response = await fetch(markdownUrl);

  if (!response.ok) {
    throw new Error(`Failed to fetch markdown: ${response.status} ${response.statusText}`);
  }

  const buffer = await response.arrayBuffer();

  return new TextDecoder("utf-8").decode(buffer);
}

async function copyAsMarkdown() {
  try {
    const text = await fetchMarkdownText(getMarkdownURL());
    await navigator.clipboard.writeText(text);

    copied.value = true;
    clearTimeout(copiedTimer);
    copiedTimer = setTimeout(() => {
      copied.value = false;
    }, COPIED_RESET_DELAY);
  } catch (error) {
    console.error("Error copying markdown:", error);
  }
}

function viewAsMarkdown() {
  window.open(getMarkdownURL(), "_blank");
}

function openInAI(provider) {
  const prompt = buildPrompt(getMarkdownURL());

  window.open(provider.promptUrl + encodeURIComponent(prompt), "_blank");
}

onUnmounted(() => {
  clearTimeout(copiedTimer);
});
</script>

<style scoped>
.markdown-copy-buttons {
  width: 100%;
  display: flex;
  margin: 10px 0 20px;
  padding-bottom: 14px;
  border-bottom: 1px solid color-mix(in srgb, var(--vp-c-divider) 50%, transparent);
}

.markdown-copy-buttons-inner {
  display: flex;
  align-items: center;
  gap: 10px;
}

.copy-page {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 30px;
  padding: 5px 10px;
  background: transparent;
  border: 1.5px solid var(--vp-c-divider);
  border-radius: 6px;
  color: var(--vp-c-text-1);
  font-size: 12px;
  font-weight: 600;
  line-height: 16px;
  white-space: nowrap;
  cursor: pointer;
}

.label {
  white-space: nowrap;
}

.icon {
  width: 16px;
  height: 16px;
  flex: 0 0 16px;
  color: var(--vp-c-text-2);
}

@media (prefers-reduced-motion: no-preference) {
  .copy-page {
    transition:
      background-color 0.16s ease,
      border-color 0.16s ease,
      color 0.16s ease;
  }

  .copy-page:hover {
    background: color-mix(in srgb, var(--vp-c-brand-1) 8%, transparent);
    border-color: color-mix(in srgb, var(--vp-c-brand-1) 40%, transparent);
    color: var(--vp-c-brand-1);
  }
}
</style>
