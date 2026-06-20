<template>
  <div class="markdown-copy-buttons">
    <div class="markdown-copy-buttons-inner">
      <div class="dropdown-container" ref="dropdownContainer">
        <button class="open-page" :aria-expanded="isOpen" @click.stop="toggleDropdown">
          <span v-html="icons['sparkle']" class="icon"></span>
          <span class="label">Ask AI</span>
          <span v-html="icons['chevron']" class="icon chevron" :class="{ open: isOpen }"></span>
        </button>

        <div v-if="isRendered" ref="dropdownMenu" class="dropdown-menu" :class="{ open: isOpen }">
          <button class="dropdown-item" @click="viewAsMarkdown">
            <span v-html="icons['markdown']" class="icon"></span>
            View as Markdown
            <span v-html="icons['external']" class="icon external"></span>
          </button>

          <button
            v-for="provider in aiProviders"
            :key="provider.name"
            class="dropdown-item"
            @click="openInAI(provider)"
          >
            <span v-html="icons[provider.icon]" class="icon"></span>
            Open in {{ provider.name }}
            <span v-html="icons['external']" class="icon external"></span>
          </button>
        </div>
      </div>
      <button class="copy-page" @click="copyAsMarkdown">
        <span v-html="icons[copied ? 'check' : 'copy']" class="icon"></span>
        <span class="label">
          {{ copied ? "Copied" : "Copy Markdown" }}
        </span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from "vue";

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
    buildPrompt: (markdownUrl) => `Read from ${markdownUrl} so I can ask questions about it.`,
  },
  {
    icon: "claude",
    name: "Claude",
    promptUrl: "https://claude.ai/new?q=",
    buildPrompt: (markdownUrl) => `Read from ${markdownUrl} so I can ask questions about it.`,
  },
  {
    icon: "kimi",
    name: "Kimi",
    promptUrl: "https://www.kimi.com/_prefill_chat?prefill_prompt=",
    buildPrompt: (markdownUrl) => `Read from ${markdownUrl} so I can ask questions about it.`,
  },
];

const isOpen = ref(false);
const copied = ref(false);
const dropdownContainer = ref();
const isRendered = ref(false);
const dropdownMenu = ref();

const animationDuration = 2000;

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

function getCurrentURL() {
  if (typeof window === "undefined") {
    return "";
  }

  return window.location.origin + normalizeLegacyZhPathname(window.location.pathname);
}

function getMarkdownURL() {
  return resolveMarkdownPageURL(getCurrentURL());
}

async function fetchMarkdownText(markdownUrl) {
  const response = await fetch(markdownUrl);

  if (!response.ok) {
    throw new Error(`Failed to fetch markdown: ${response.status} ${response.statusText}`);
  }

  const buffer = await response.arrayBuffer();

  return new TextDecoder("utf-8").decode(buffer);
}

function toggleDropdown() {
  if (isOpen.value) {
    isOpen.value = false;

    const el = dropdownMenu.value;
    if (!el) {
      return;
    }

    const onEnd = () => {
      isRendered.value = false;
      el.removeEventListener("transitionend", onEnd);
    };

    el.addEventListener("transitionend", onEnd);
  } else {
    isRendered.value = true;
    requestAnimationFrame(() => {
      isOpen.value = true;
    });
  }
}

async function copyAsMarkdown() {
  try {
    const text = await fetchMarkdownText(getMarkdownURL());
    await navigator.clipboard.writeText(text);

    copied.value = true;
    setTimeout(() => {
      copied.value = false;
    }, animationDuration);
  } catch (error) {
    console.error("Error copying markdown:", error);
  }

  isOpen.value = false;
}

async function viewAsMarkdown() {
  const markdownUrl = getMarkdownURL();
  window.open(markdownUrl, "_blank");
  isOpen.value = false;
}

function openInAI(provider) {
  const markdownUrl = getMarkdownURL();
  const prompt = provider.buildPrompt(markdownUrl);

  window.open(provider.promptUrl + encodeURIComponent(prompt), "_blank");
  isOpen.value = false;
}

function handleClickOutside(event) {
  if (dropdownContainer.value && !dropdownContainer.value.contains(event.target)) {
    isOpen.value = false;
  }
}

onMounted(() => document.addEventListener("click", handleClickOutside));
onUnmounted(() => document.removeEventListener("click", handleClickOutside));
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
  position: relative;
}

.dropdown-container {
  position: relative;
}

.copy-page,
.open-page {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 30px;
  padding: 5px 10px;
  background: transparent;
  border: 1.5px solid color-mix(in srgb, var(--vp-c-divider) 72%, transparent);
  border-radius: 6px;
  color: var(--vp-c-text-2);
  font-size: 12px;
  font-weight: 600;
  line-height: 16px;
  white-space: nowrap;
  cursor: pointer;
}

.label {
  white-space: nowrap;
}

.dropdown-menu {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  width: 222px;
  padding: 4px;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  z-index: 30;
  box-shadow:
    0 10px 24px rgba(0, 0, 0, 0.12),
    0 2px 8px rgba(0, 0, 0, 0.08);
  opacity: 0;
  transform: translateY(-4px) scale(0.98);
  pointer-events: none;
}

.dropdown-menu.open {
  opacity: 1;
  transform: translateY(0) scale(1);
  pointer-events: auto;
}

.dropdown-item {
  position: relative;
  width: 100%;
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 36px;
  padding: 8px;
  background: transparent;
  border: none;
  border-radius: 8px;
  color: var(--vp-c-text-1);
  font-size: 14px;
  line-height: 20px;
  cursor: pointer;
  text-align: left;
}

.dropdown-item .icon.external {
  margin-left: auto;
  opacity: 0.6;
}

.icon {
  width: 16px;
  height: 16px;
  flex: 0 0 16px;
  color: var(--vp-c-text-2);
}

.dropdown-item .icon {
  width: 18px;
  height: 18px;
  flex-basis: 18px;
}

.chevron.open {
  transform: rotate(180deg);
}

.dropdown-item:hover .icon.external {
  opacity: 1;
  transform: translateX(2px);
}

@media (prefers-reduced-motion: no-preference) {
  .dropdown-menu {
    transition:
      opacity 0.18s cubic-bezier(0.4, 0, 0.2, 1),
      transform 0.18s cubic-bezier(0.4, 0, 0.2, 1);
    transform-origin: top;
  }

  .copy-page,
  .open-page,
  .dropdown-item,
  .dropdown-item .icon.external {
    transition:
      background-color 0.16s ease,
      border-color 0.16s ease,
      color 0.16s ease,
      transform 0.16s ease,
      opacity 0.16s ease;
  }

  .copy-page:hover,
  .open-page:hover {
    background: color-mix(in srgb, var(--vp-c-brand-1) 8%, transparent);
    border-color: color-mix(in srgb, var(--vp-c-brand-1) 40%, transparent);
    color: var(--vp-c-brand-1);
  }

  .dropdown-item:hover {
    background: var(--vp-c-bg-soft);
  }

  .chevron {
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }
}
</style>
