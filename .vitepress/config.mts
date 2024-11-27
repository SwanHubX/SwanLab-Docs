import { defineConfig } from 'vitepress'
import { zh } from './zh'
import { en } from './en'

export default defineConfig({
  // 标签页logo
  head: [
    ['link', { rel: 'icon', type:"image/svg+xml", href: '/icon.svg' }],
    ['link', { rel: 'icon', type:"image/png", href: '/icon.png' }],
  ],
  locales: {
    zh: { label: '简体中文', ...zh },
    en: { label: 'English', ...en },
  }
})