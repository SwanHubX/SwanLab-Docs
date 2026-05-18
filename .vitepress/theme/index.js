// .vitepress/theme/index.js
import DefaultTheme from 'vitepress/theme'
import mediumZoom from 'medium-zoom'
import { onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'
import './custom.css'
import HeaderButton from './components/HeaderButton.vue'
import HeaderButtonEN from './components/HeaderButtonEN.vue'
import HeaderGithubButton from './components/HeaderGithubButton.vue'
import HeaderDocHelperButton from './components/HeaderDocHelperButton.vue'
import HeaderDocHelperButtonEN from './components/HeaderDocHelperButtonEN.vue'

export default {
  ...DefaultTheme,
  enhanceApp({ app }) {
    app.component('HeaderButton', HeaderButton)
    app.component('HeaderButtonEN', HeaderButtonEN)
    app.component('HeaderGithubButton', HeaderGithubButton)
    app.component('HeaderDocHelperButton', HeaderDocHelperButton)
    app.component('HeaderDocHelperButtonEN', HeaderDocHelperButtonEN)
    DefaultTheme.enhanceApp({ app })
  },
  setup() {
    const route = useRoute()

    const shouldZoomImage = (img) => {
      // 1. Check if there are clear exclusion marks
      if (img.classList.contains('no-zoomable') || img.hasAttribute('data-no-zoom') || img.closest('a[href]')) {
        return false
      }

      // 2. Exclude specified directories
      const src = img.src || img.getAttribute('src') || ''
      const excludedDirectories = ['/exclude/']
      if (excludedDirectories.some(dir => src.includes(dir))) {
        return false
      }

      return true
    }

    // Image zoom functionality
    const initZoom = () => {
      nextTick(() => {
        const allImages = document.querySelectorAll('.vp-doc img')
        const zoomableImages = Array.from(allImages).filter(shouldZoomImage)

        if (zoomableImages.length > 0) {
          mediumZoom(zoomableImages, {
            background: 'var(--vp-c-bg)',
          })
        }
      })
    }

    onMounted(() => {
      initZoom()
    })

    watch(
      () => route.path,
      () => nextTick(() => initZoom())
    )
  }
}
