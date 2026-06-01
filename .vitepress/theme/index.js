// .vitepress/theme/index.js
import DefaultTheme from 'vitepress/theme'
import mediumZoom from 'medium-zoom'
import { onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'
import './custom.css'
import HeaderButton from './components/HeaderButton.vue'
import HeaderButtonEN from './components/HeaderButtonEN.vue'
import HeaderGithubButton from './components/HeaderGithubButton.vue'
import HeaderDocHelperButton from './components/HeaderDocHelperButton.vue'
import HeaderDocHelperButtonEN from './components/HeaderDocHelperButtonEN.vue'
import CopyOrDownloadAsMarkdownButtons from './components/CopyOrDownloadAsMarkdownButtons.vue'

export default {
  ...DefaultTheme,
  enhanceApp({ app }) {
    app.component('HeaderButton', HeaderButton)
    app.component('HeaderButtonEN', HeaderButtonEN)
    app.component('HeaderGithubButton', HeaderGithubButton)
    app.component('HeaderDocHelperButton', HeaderDocHelperButton)
    app.component('HeaderDocHelperButtonEN', HeaderDocHelperButtonEN)
    app.component('CopyOrDownloadAsMarkdownButtons', CopyOrDownloadAsMarkdownButtons)
    DefaultTheme.enhanceApp({ app })
  },
  setup() {
    const route = useRoute()
    let navSearchPlaceholder
    let navSearchAnimationFrame = 0

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

    const arrangeNavSearch = () => {
      if (typeof window === 'undefined') {
        return
      }

      const contentBody = document.querySelector('.VPNavBar .content-body')
      const menu = document.querySelector('.VPNavBarMenu.menu')
      const search = document.querySelector('.VPNavBarSearch.search')

      if (!contentBody || !menu || !search) {
        return
      }

      if (!navSearchPlaceholder) {
        navSearchPlaceholder = document.createComment('swanlab-nav-search-position')
        const placeholderTarget = search.parentElement === contentBody ? search : menu
        contentBody.insertBefore(navSearchPlaceholder, placeholderTarget)
      }

      if (window.innerWidth >= 1280) {
        const firstRightAction = menu.querySelector('.vp-header-doc-helper-btn, .header-button, .github-button')

        if (firstRightAction) {
          menu.insertBefore(search, firstRightAction)
        }

        return
      }

      if (navSearchPlaceholder.parentNode) {
        navSearchPlaceholder.parentNode.insertBefore(search, navSearchPlaceholder.nextSibling)
      }
    }

    const scheduleNavSearchArrangement = () => {
      if (typeof window === 'undefined') {
        return
      }

      window.cancelAnimationFrame(navSearchAnimationFrame)
      navSearchAnimationFrame = window.requestAnimationFrame(arrangeNavSearch)
    }

    onMounted(() => {
      initZoom()
      nextTick(() => {
        arrangeNavSearch()
        window.addEventListener('resize', scheduleNavSearchArrangement)
      })
    })

    onUnmounted(() => {
      if (typeof window === 'undefined') {
        return
      }

      window.removeEventListener('resize', scheduleNavSearchArrangement)
      window.cancelAnimationFrame(navSearchAnimationFrame)
    })

    watch(
      () => route.path,
      () => nextTick(() => {
        initZoom()
        arrangeNavSearch()
      })
    )
  }
}
