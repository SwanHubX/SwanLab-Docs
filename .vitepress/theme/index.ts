// .vitepress/theme/index.ts
import type { Theme } from "vitepress";
import { useRoute } from "vitepress";
import { VoidZeroTheme, themeContextKey } from "@voidzero-dev/vitepress-theme";
import { onMounted, watch, nextTick } from "vue";
import "virtual:group-icons.css";
// Theme design system first; project overrides after (cascade: tokens → layout → components).
import "./styles.css";
import "./styles/tokens.css";
import "./styles/layout.css";
import "./styles/components.css";
import Layout from "./Layout.vue";
import HeaderButton from "./components/HeaderButton.vue";
import HeaderButtonEN from "./components/HeaderButtonEN.vue";
import HeaderGithubButton from "./components/HeaderGithubButton.vue";
// Deprecated: Docs Copilot / 文档助手 is offline; keep components registered for temporary rollback.
import HeaderDocHelperButton from "./components/HeaderDocHelperButton.vue";
import HeaderDocHelperButtonEN from "./components/HeaderDocHelperButtonEN.vue";
import CopyOrDownloadAsMarkdownButtons from "./components/CopyOrDownloadAsMarkdownButtons.vue";
import VersionBadge from "./components/VersionBadge.vue";
import BadgeRow from "./components/BadgeRow.vue";
import MermaidDiagram from "./components/MermaidDiagram.vue";
// 首页合作推广区（可选组件，组件内 enabled=false 即可整区下线）
import HomePromoSection from "./components/HomePromoSection.vue";
import { useVersionSync } from "./version-sync";
import { useLogoLinkFix } from "./logo-link";

export default {
  ...VoidZeroTheme,
  Layout,
  enhanceApp(ctx) {
    // Required by the theme's forked header/banner/footer components, which
    // inject this context (see the theme's per-project variant entries).
    // Placeholder assets for now — brand polish comes later.
    ctx.app.provide(themeContextKey, {
      // logoDark is rendered in light mode (dark-colored logo on light bg),
      // logoLight in dark mode — see OSS Header.vue (`block dark:hidden` / `dark:block`).
      logoDark: "/logo_2.svg",
      logoLight: "/logo_dark_2.svg",
      logoAlt: "SwanLab",
      // 1px transparent gif so the OSS footer CTA renders without a photo.
      footerBg: "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA=",
      monoIcon: "/logo_2.svg",
    });
    ctx.app.component("HeaderButton", HeaderButton);
    ctx.app.component("HeaderButtonEN", HeaderButtonEN);
    ctx.app.component("HeaderGithubButton", HeaderGithubButton);
    // Deprecated: not used in nav while Docs Copilot / 文档助手 is offline.
    ctx.app.component("HeaderDocHelperButton", HeaderDocHelperButton);
    ctx.app.component("HeaderDocHelperButtonEN", HeaderDocHelperButtonEN);
    ctx.app.component("CopyOrDownloadAsMarkdownButtons", CopyOrDownloadAsMarkdownButtons);
    ctx.app.component("VersionBadge", VersionBadge);
    ctx.app.component("BadgeRow", BadgeRow);
    ctx.app.component("MermaidDiagram", MermaidDiagram);
    ctx.app.component("HomePromoSection", HomePromoSection);
    VoidZeroTheme.enhanceApp(ctx);
  },
  setup() {
    const route = useRoute();
    const initVersionSync = useVersionSync();
    const initLogoLinkFix = useLogoLinkFix();

    const shouldZoomImage = (img: HTMLImageElement) => {
      // 1. Check if there are clear exclusion marks
      if (
        img.classList.contains("no-zoomable") ||
        img.hasAttribute("data-no-zoom") ||
        img.closest("a[href]")
      ) {
        return false;
      }

      // 2. Exclude specified directories
      const src = img.src || img.getAttribute("src") || "";
      const excludedDirectories = ["/exclude/"];
      if (excludedDirectories.some((dir) => src.includes(dir))) {
        return false;
      }

      return true;
    };

    // Image zoom functionality
    const initZoom = () => {
      void nextTick(() => {
        const allImages = document.querySelectorAll<HTMLImageElement>(".vp-doc img");
        const zoomableImages = Array.from(allImages).filter(shouldZoomImage);

        if (zoomableImages.length === 0) return;

        // Lazy-load medium-zoom only when a zoomable image exists,
        // keeping it out of the critical-path theme chunk.
        void import("medium-zoom").then(({ default: mediumZoom }) => {
          mediumZoom(zoomableImages, {
            background: "var(--vp-c-bg)",
          });
        });
      });
    };

    onMounted(() => {
      initZoom();
      // Browser-side navbar version sync (localStorage 6h TTL + silent PyPI fetch).
      initVersionSync();
      // Locale-aware navbar logo link (theme hardcodes href="/"; en → /en/).
      initLogoLinkFix();
    });

    watch(
      () => route.path,
      () =>
        void nextTick(() => {
          initZoom();
        }),
    );
  },
} satisfies Theme;
