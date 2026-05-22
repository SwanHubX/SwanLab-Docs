<template>
  <div class="markdown-copy-buttons">
    <div class="markdown-copy-buttons-inner">
      <div class="dropdown-container" ref="dropdownContainer">
        <button class="open-page" :aria-expanded="isOpen" @click.stop="toggleDropdown">
          <span v-html="iconSparkle" class="icon"></span>
          <span class="label">Ask AI</span>
          <span v-html="iconChevron" class="icon chevron" :class="{ open: isOpen }"></span>
        </button>

        <div v-if="isRendered" ref="dropdownMenu" class="dropdown-menu" :class="{ open: isOpen }">
          <button class="dropdown-item" @click="viewAsMarkdown">
            <span v-html="iconMarkdown" class="icon"></span>
            View as Markdown
            <span v-html="iconExternal" class="icon external"></span>
          </button>

          <button
            v-for="provider in aiProviders"
            :key="provider.name"
            class="dropdown-item"
            @click="openInAI(provider)"
          >
            <span v-html="provider.icon" class="icon"></span>
            Open in {{ provider.name }}
            <span v-html="iconExternal" class="icon external"></span>
          </button>
        </div>
      </div>
      <button class="copy-page" @click="copyAsMarkdown">
        <span v-html="copied ? iconCheck : iconCopy" class="icon"></span>
        <span class="label">
          {{ copied ? 'Copied' : 'Copy Markdown' }}
        </span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue'

const iconChatGPT =
  '<svg viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91 6.046 6.046 0 0 0-6.51-2.9A6.065 6.065 0 0 0 4.981 4.18a5.985 5.985 0 0 0-3.998 2.9 6.046 6.046 0 0 0 .743 7.097 5.98 5.98 0 0 0 .51 4.911 6.051 6.051 0 0 0 6.515 2.9A5.985 5.985 0 0 0 13.26 24a6.056 6.056 0 0 0 5.772-4.206 5.99 5.99 0 0 0 3.997-2.9 6.056 6.056 0 0 0-.747-7.073zM13.26 22.43a4.476 4.476 0 0 1-2.876-1.04l.141-.081 4.779-2.758a.795.795 0 0 0 .392-.681v-6.737l2.02 1.168a.071.071 0 0 1 .038.052v5.583a4.504 4.504 0 0 1-4.494 4.494zM3.6 18.304a4.47 4.47 0 0 1-.535-3.014l.142.085 4.783 2.759a.771.771 0 0 0 .78 0l5.843-3.369v2.332a.08.08 0 0 1-.033.062L9.74 19.95a4.5 4.5 0 0 1-6.14-1.646zM2.34 7.896a4.485 4.485 0 0 1 2.366-1.973V11.6a.766.766 0 0 0 .388.676l5.815 3.355-2.02 1.168a.076.076 0 0 1-.071 0l-4.83-2.786A4.504 4.504 0 0 1 2.34 7.872zm16.597 3.855l-5.833-3.387L15.119 7.2a.076.076 0 0 1 .071 0l4.83 2.791a4.494 4.494 0 0 1-.676 8.105v-5.678a.79.79 0 0 0-.407-.667zm2.01-3.023l-.141-.085-4.774-2.782a.776.776 0 0 0-.785 0L9.409 9.23V6.897a.066.066 0 0 1 .028-.061l4.83-2.787a4.5 4.5 0 0 1 6.68 4.66zm-12.64 4.135l-2.02-1.164a.08.08 0 0 1-.038-.057V6.075a4.5 4.5 0 0 1 7.375-3.453l-.142.08L8.704 5.46a.795.795 0 0 0-.393.681zm1.097-2.365l2.602-1.5 2.607 1.5v2.999l-2.597 1.5-2.607-1.5z"/></svg>'
const iconCheck =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="m20 6-11 11-5-5"/></svg>'
const iconChevron =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="m6 9 6 6 6-6"/></svg>'
const iconClaude =
  '<svg viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M11.376 24L10.776 23.544L10.44 22.8L10.776 21.312L11.16 19.392L11.472 17.856L11.76 15.96L11.928 15.336L11.904 15.288L11.784 15.312L10.344 17.28L8.16 20.232L6.432 22.056L6.024 22.224L5.304 21.864L5.376 21.192L5.784 20.616L8.16 17.568L9.6 15.672L10.536 14.592L10.512 14.448H10.464L4.128 18.576L3 18.72L2.496 18.264L2.568 17.52L2.808 17.28L4.704 15.96L9.432 13.32L9.504 13.08L9.432 12.96H9.192L8.4 12.912L5.712 12.84L3.384 12.744L1.104 12.624L0.528 12.504L0 11.784L0.048 11.424L0.528 11.112L1.224 11.16L2.736 11.28L5.016 11.424L6.672 11.52L9.12 11.784H9.504L9.552 11.616L9.432 11.52L9.336 11.424L6.96 9.84L4.416 8.16L3.072 7.176L2.352 6.672L1.992 6.216L1.848 5.208L2.496 4.488L3.384 4.56L3.6 4.608L4.488 5.304L6.384 6.768L8.88 8.616L9.24 8.904L9.408 8.808V8.736L9.24 8.472L7.896 6.024L6.456 3.528L5.808 2.496L5.64 1.872C5.576 1.656 5.544 1.416 5.544 1.152L6.288 0.144001L6.696 0L7.704 0.144001L8.112 0.504001L8.736 1.92L9.72 4.152L11.28 7.176L11.736 8.088L11.976 8.904L12.072 9.168H12.24V9.024L12.36 7.296L12.6 5.208L12.84 2.52L12.912 1.752L13.296 0.840001L14.04 0.360001L14.616 0.624001L15.096 1.32L15.024 1.752L14.76 3.6L14.184 6.504L13.824 8.472H14.04L14.28 8.208L15.264 6.912L16.92 4.848L17.64 4.032L18.504 3.12L19.056 2.688H20.088L20.832 3.816L20.496 4.992L19.44 6.336L18.552 7.464L17.28 9.168L16.512 10.536L16.584 10.632H16.752L19.608 10.008L21.168 9.744L22.992 9.432L23.832 9.816L23.928 10.2L23.592 11.016L21.624 11.496L19.32 11.952L15.888 12.768L15.84 12.792L15.888 12.864L17.424 13.008L18.096 13.056H19.728L22.752 13.272L23.544 13.8L24 14.424L23.928 14.928L22.704 15.528L21.072 15.144L17.232 14.232L15.936 13.92H15.744V14.016L16.848 15.096L18.84 16.896L21.36 19.224L21.48 19.8L21.168 20.28L20.832 20.232L18.624 18.552L17.76 17.808L15.84 16.2H15.72V16.368L16.152 17.016L18.504 20.544L18.624 21.624L18.456 21.96L17.832 22.176L17.184 22.056L15.792 20.136L14.376 17.952L13.224 16.008L13.104 16.104L12.408 23.352L12.096 23.712L11.376 24Z" shape-rendering="optimizeQuality"/></svg>'
const iconCopy =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>'
const iconExternal =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>'
const iconMarkdown =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 15V9l3 3 3-3v6"/><path d="m17 9 2 3-2 3"/><path d="M19 12h-4"/></svg>'
const iconKimi =
  '<svg version="1.1" role="img" viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg"><path d="M842.1632 246.4512c6.9376-8.9344 13.056-17.0752 19.4816-24.96 2.9952-3.712 2.7392-6.528-0.1536-10.4192-27.9552-36.736-30.592-77.5168-14.5152-118.912 12.0832-31.1552 38.784-45.7472 71.424-48.8448 20.352-1.92 40.32 0.1536 58.8288 10.0608 24.32 13.0048 38.5024 32.8448 43.1104 60.2368 3.6608 21.8624 2.9696 43.1872-3.2 64.3584-10.9824 37.4528-37.888 56.8576-74.8032 61.7728-30.6432 4.096-61.696 4.608-92.5952 6.7072-2.3808 0.1536-4.8128 0-7.5776 0z" fill="#027AFF"></path><path d="M766.3872 78.6688h-184.576L435.6608 411.904h-206.592V80.128H64v858.5472h165.12V576.9728h291.1488a129.0752 129.0752 0 0 0 117.0432-74.6496v436.352h165.12V576.9728a165.12 165.12 0 0 0-153.088-164.6848v-0.4352h-90.6752a168.1152 168.1152 0 0 0 99.1232-90.4448l108.5952-242.7392z" fill="#000000"></path></svg>'
const iconSparkle =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/></svg>'

const aiProviders = [
  {
    icon: iconChatGPT,
    name: 'ChatGPT',
    promptUrl: 'https://chatgpt.com/?hints=search&prompt=',
    buildPrompt: (markdownUrl) => `Read from ${markdownUrl} so I can ask questions about it.`,
  },
  {
    icon: iconClaude,
    name: 'Claude',
    promptUrl: 'https://claude.ai/new?q=',
    buildPrompt: (markdownUrl) => `Read from ${markdownUrl} so I can ask questions about it.`,
  },
  {
    icon: iconKimi,
    name: 'Kimi',
    promptUrl: 'https://www.kimi.com/_prefill_chat?prefill_prompt=',
    buildPrompt: (markdownUrl) => `Read from ${markdownUrl} so I can ask questions about it.`,
  },
]

const isOpen = ref(false)
const copied = ref(false)
const dropdownContainer = ref()
const isRendered = ref(false)
const dropdownMenu = ref()

const animationDuration = 2000

function removeHtmlExtension(pathSegment) {
  const lastSlashIndex = pathSegment.lastIndexOf('/')
  const lastDotIndex = pathSegment.lastIndexOf('.')

  if (lastDotIndex > lastSlashIndex && lastDotIndex !== -1 && pathSegment.endsWith('.html')) {
    return pathSegment.slice(0, lastDotIndex)
  }

  return pathSegment
}

function cleanUrl(url) {
  const { origin, pathname } = new URL(url)
  const pathnameWithoutTrailingSlash = pathname.replace(/\/+$/, '')

  if (pathname.length > 0) {
    return origin + removeHtmlExtension(pathnameWithoutTrailingSlash)
  }

  return origin
}

function resolveMarkdownPageURL(url) {
  const cleanedURL = cleanUrl(url)

  if (cleanedURL === window.location.origin) {
    return `${cleanedURL}/index.md`
  }

  return `${cleanedURL}.md`
}

function normalizeLegacyZhPathname(pathname) {
  if (pathname === '/zh') {
    return '/'
  }

  if (pathname.startsWith('/zh/')) {
    return pathname.slice(3)
  }

  return pathname
}

function getCurrentURL() {
  if (typeof window === 'undefined') {
    return ''
  }

  return window.location.origin + normalizeLegacyZhPathname(window.location.pathname)
}

function getMarkdownURL() {
  return resolveMarkdownPageURL(getCurrentURL())
}

async function fetchMarkdownText(markdownUrl) {
  const response = await fetch(markdownUrl)

  if (!response.ok) {
    throw new Error(`Failed to fetch markdown: ${response.status} ${response.statusText}`)
  }

  const buffer = await response.arrayBuffer()

  return new TextDecoder('utf-8').decode(buffer)
}

function getMarkdownFilename(markdownUrl) {
  const { pathname } = new URL(markdownUrl)

  return decodeURIComponent(pathname.split('/').pop() || 'page.md')
}

function escapeHTML(value) {
  return value.replace(/[&<>]/g, (char) => {
    if (char === '&') {
      return '&amp;'
    }

    if (char === '<') {
      return '&lt;'
    }

    return '&gt;'
  })
}

function renderMarkdownViewer(viewer, title, markdownText) {
  const escapedTitle = escapeHTML(title)
  const escapedMarkdown = escapeHTML(markdownText)

  viewer.document.open()
  viewer.document.write(`<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapedTitle}</title>
  <style>
    :root {
      color-scheme: light dark;
      background: Canvas;
      color: CanvasText;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }

    body {
      margin: 0;
    }

    pre {
      box-sizing: border-box;
      min-height: 100vh;
      margin: 0;
      padding: 24px;
      font: inherit;
      font-size: 14px;
      line-height: 1.7;
      overflow-wrap: break-word;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <pre>${escapedMarkdown}</pre>
</body>
</html>`)
  viewer.document.close()
}

function toggleDropdown() {
  if (isOpen.value) {
    isOpen.value = false

    const el = dropdownMenu.value
    if (!el) {
      return
    }

    const onEnd = () => {
      isRendered.value = false
      el.removeEventListener('transitionend', onEnd)
    }

    el.addEventListener('transitionend', onEnd)
  } else {
    isRendered.value = true
    requestAnimationFrame(() => {
      isOpen.value = true
    })
  }
}

async function copyAsMarkdown() {
  try {
    const text = await fetchMarkdownText(getMarkdownURL())
    await navigator.clipboard.writeText(text)

    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, animationDuration)
  } catch (error) {
    console.error('Error copying markdown:', error)
  }

  isOpen.value = false
}

async function viewAsMarkdown() {
  const markdownUrl = getMarkdownURL()
  const viewer = window.open('', '_blank')
  isOpen.value = false

  if (!viewer) {
    return
  }

  const filename = getMarkdownFilename(markdownUrl)
  viewer.opener = null
  renderMarkdownViewer(viewer, filename, 'Loading Markdown...')

  try {
    const text = await fetchMarkdownText(markdownUrl)
    renderMarkdownViewer(viewer, filename, text)
  } catch (error) {
    console.error('Error viewing markdown:', error)
    renderMarkdownViewer(viewer, filename, `Unable to load Markdown.\n\n${String(error)}`)
  }
}

function openInAI(provider) {
  const markdownUrl = getMarkdownURL()
  const prompt = provider.buildPrompt(markdownUrl)

  window.open(provider.promptUrl + encodeURIComponent(prompt), '_blank')
  isOpen.value = false
}

function handleClickOutside(event) {
  if (dropdownContainer.value && !dropdownContainer.value.contains(event.target)) {
    isOpen.value = false
  }
}

onMounted(() => document.addEventListener('click', handleClickOutside))
onUnmounted(() => document.removeEventListener('click', handleClickOutside))
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
