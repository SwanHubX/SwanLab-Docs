<template>
  <div class="markdown-copy-buttons">
    <div class="markdown-copy-buttons-inner">
      <div class="dropdown-container" ref="dropdownContainer">
        <div class="dropdown-trigger">
          <button class="copy-page" @click="copyAsMarkdown">
            <span v-html="copied ? iconCheck : iconCopy" class="icon"></span>
            <span class="label">
              {{ copied ? 'Copied' : 'Copy page' }}
            </span>
          </button>

          <span class="divider"></span>

          <button class="chevron-wrapper" @click.stop="toggleDropdown">
            <span v-html="iconChevron" class="icon chevron" :class="{ open: isOpen }"></span>
          </button>
        </div>

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

      <button class="download-btn" @click="downloadMarkdown">
        <span v-html="downloaded ? iconCheck : iconDownload" class="icon"></span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue'

const iconChatGPT =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="M12 3.5a4 4 0 0 1 6.9 2.7 4 4 0 0 1-.2 7.4 4 4 0 0 1-5.8 5.3 4 4 0 0 1-6.9-2.7 4 4 0 0 1 .2-7.4A4 4 0 0 1 12 3.5Z"/><path d="M8.5 8.8 12 6.8l3.5 2"/><path d="M15.5 8.8v4L12 15l-3.5-2.2v-4"/><path d="m8.5 12.8 3.5 2.2 3.5-2.2"/></svg>'
const iconCheck =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="m20 6-11 11-5-5"/></svg>'
const iconChevron =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="m6 9 6 6 6-6"/></svg>'
const iconClaude =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="M12 3.5 4.5 19.5"/><path d="m12 3.5 7.5 16"/><path d="M7 14h10"/><path d="M9.6 8.4 14.4 19.5"/></svg>'
const iconCopy =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>'
const iconDownload =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/></svg>'
const iconExternal =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>'
const iconMarkdown =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 15V9l3 3 3-3v6"/><path d="m17 9 2 3-2 3"/><path d="M19 12h-4"/></svg>'
const iconKimi =
  '<svg version="1.1" role="img" viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg"><path d="M842.1632 246.4512c6.9376-8.9344 13.056-17.0752 19.4816-24.96 2.9952-3.712 2.7392-6.528-0.1536-10.4192-27.9552-36.736-30.592-77.5168-14.5152-118.912 12.0832-31.1552 38.784-45.7472 71.424-48.8448 20.352-1.92 40.32 0.1536 58.8288 10.0608 24.32 13.0048 38.5024 32.8448 43.1104 60.2368 3.6608 21.8624 2.9696 43.1872-3.2 64.3584-10.9824 37.4528-37.888 56.8576-74.8032 61.7728-30.6432 4.096-61.696 4.608-92.5952 6.7072-2.3808 0.1536-4.8128 0-7.5776 0z" fill="#027AFF"></path><path d="M766.3872 78.6688h-184.576L435.6608 411.904h-206.592V80.128H64v858.5472h165.12V576.9728h291.1488a129.0752 129.0752 0 0 0 117.0432-74.6496v436.352h165.12V576.9728a165.12 165.12 0 0 0-153.088-164.6848v-0.4352h-90.6752a168.1152 168.1152 0 0 0 99.1232-90.4448l108.5952-242.7392z" fill="#000000"></path></svg>'

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
const downloaded = ref(false)
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

function getCurrentURL() {
  if (typeof window === 'undefined') {
    return ''
  }

  return window.location.origin + window.location.pathname
}

function getMarkdownURL() {
  return resolveMarkdownPageURL(getCurrentURL())
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
    const response = await fetch(getMarkdownURL())
    const text = await response.text()
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

function viewAsMarkdown() {
  window.open(getMarkdownURL(), '_blank')
  isOpen.value = false
}

function openInAI(provider) {
  const markdownUrl = getMarkdownURL()
  const prompt = provider.buildPrompt(markdownUrl)

  window.open(provider.promptUrl + encodeURIComponent(prompt), '_blank')
  isOpen.value = false
}

function downloadFile(filename, content, blobType = 'text/plain') {
  const blob = content instanceof Blob ? content : new Blob([content], { type: blobType })
  const url = URL.createObjectURL(blob)

  Object.assign(document.createElement('a'), {
    download: filename,
    href: url,
  }).click()

  URL.revokeObjectURL(url)
}

async function downloadMarkdown() {
  try {
    const markdownUrl = getMarkdownURL()
    const response = await fetch(markdownUrl)
    const text = await response.text()
    const filename = markdownUrl.split('/').pop() || 'page.md'

    downloadFile(filename, text, 'text/markdown')
    downloaded.value = true
    setTimeout(() => {
      downloaded.value = false
    }, animationDuration)
  } catch (error) {
    console.error('Error downloading markdown:', error)
  }
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
  margin-bottom: 16px;
}

.markdown-copy-buttons-inner {
  margin: 16px 0;
  display: flex;
  gap: 8px;
  position: relative;
}

.dropdown-container {
  position: relative;
}

.dropdown-trigger {
  display: flex;
  align-items: stretch;
  background: transparent;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  color: var(--vp-c-text-1);
  font-size: 14px;
  padding: 0;
  overflow: hidden;
}

.copy-page {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  cursor: pointer;
  white-space: nowrap;
  background: transparent;
  border: none;
  color: inherit;
}

.label {
  white-space: nowrap;
}

.divider {
  width: 1px;
  height: 25px;
  align-self: center;
  background: var(--vp-c-divider);
  opacity: 0.6;
}

.chevron-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 12px;
  cursor: pointer;
  background: transparent;
  border: none;
  color: inherit;
}

.dropdown-menu {
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  min-width: 240px;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
  z-index: 100;
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
  opacity: 0;
  transform: translateY(-6px) scale(0.96);
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
  gap: 10px;
  padding: 10px 16px;
  background: transparent;
  border: none;
  color: var(--vp-c-text-1);
  font-size: 14px;
  cursor: pointer;
  text-align: left;
}

.dropdown-item .icon.external {
  margin-left: auto;
  opacity: 0.6;
}

.download-btn {
  display: flex;
  align-items: center;
  padding: 8px 12px;
  background: transparent;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  color: var(--vp-c-text-1);
  cursor: pointer;
}

.icon {
  width: 18px;
  height: 18px;
  flex: 0 0 18px;
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

  .copy-page:hover,
  .chevron-wrapper:hover,
  .download-btn:hover {
    background: var(--vp-c-bg-soft);
  }

  .dropdown-trigger,
  .copy-page,
  .chevron-wrapper,
  .dropdown-item,
  .dropdown-item .icon.external,
  .download-btn {
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .dropdown-trigger:hover,
  .download-btn:hover {
    border-color: var(--vp-c-brand-1);
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  .dropdown-item::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    width: 0;
    height: 100%;
    background: var(--vp-c-brand-1);
    transition: width 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .dropdown-item:hover {
    padding-left: 20px;
  }

  .dropdown-item:hover::before {
    width: 3px;
  }

  .chevron {
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }
}
</style>
