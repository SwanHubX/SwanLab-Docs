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
import BlogPage from './components/BlogPage.vue'
import BlogPageEN from './components/BlogPageEN.vue'

export default {
  ...DefaultTheme,
  enhanceApp({ app }) {
    app.component('HeaderButton', HeaderButton)
    app.component('HeaderButtonEN', HeaderButtonEN)
    app.component('HeaderGithubButton', HeaderGithubButton)
    app.component('HeaderDocHelperButton', HeaderDocHelperButton)
    app.component('HeaderDocHelperButtonEN', HeaderDocHelperButtonEN)
    app.component('BlogPage', BlogPage)
    app.component('BlogPageEN', BlogPageEN)
    DefaultTheme.enhanceApp({ app })
  },
  // setup() {
  //   const route = useRoute()

  //   // Image zoom functionality
  //   const initZoom = () => {
  //     mediumZoom('.main img:not(.no-zoomable)', {
  //       background: 'var(--vp-c-bg)',
  //     })
  //   }

  //   onMounted(() => {
  //     initZoom()
  //   })

  //   watch(
  //     () => route.path,
  //     () => nextTick(() => initZoom())
  //   )
  // }
}
