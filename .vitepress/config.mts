import { defineConfig } from 'vitepress'
import { zh } from './zh'
import { en } from './en'

export default defineConfig({
  rewrites: {
    'zh/:rest*': ':rest*'
  },

  themeConfig:{
    search: {
      provider: 'local'
    }
  },

  locales: {
    root: { label: '简体中文', ...zh },
    en: { label: 'English', ...en },
  }
})