// .vitepress/theme/index.ts
import "vitepress-openapi/dist/style.css";
import type { Theme } from "vitepress";
import { useData, useRoute } from "vitepress";
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
import RestApiExportButtons from "./components/RestApiExportButtons.vue";
import VersionBadge from "./components/VersionBadge.vue";
import BadgeRow from "./components/BadgeRow.vue";
import MermaidDiagram from "./components/MermaidDiagram.vue";
// 首页合作推广区（可选组件，组件内 enabled=false 即可整区下线）
import HomePromoSection from "./components/HomePromoSection.vue";
import { useVersionSync } from "./version-sync";
import { useLogoLinkFix } from "./logo-link";
// vitepress-openapi：渲染 /api/rest/ 下的 REST 接口页（mock spec）
import {
  theme as openapiTheme,
  useOpenapi,
  useTheme as useOpenapiTheme,
} from "vitepress-openapi/client";
import restApiSpec from "../specs/rest.zh.json" with { type: "json" };
import restApiSpecEn from "../specs/rest.en.json" with { type: "json" };
import { createCodeSampleGenerator } from "../utils/openapi-code-samples";

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
    ctx.app.component("RestApiExportButtons", RestApiExportButtons);
    ctx.app.component("VersionBadge", VersionBadge);
    ctx.app.component("BadgeRow", BadgeRow);
    ctx.app.component("MermaidDiagram", MermaidDiagram);
    ctx.app.component("HomePromoSection", HomePromoSection);
    VoidZeroTheme.enhanceApp(ctx);

    // vitepress-openapi：加载 mock spec，隐藏 Try it 控制台与品牌页脚
    useOpenapi({ spec: restApiSpec });
    useOpenapiTheme({
      operation: {
        hiddenSlots: ["playground", "branding"],
        cols: 1,
      },
      // 接口页会把选中的 server 持久化到 localStorage；默认 Base URL 改成
      // 私有化域名后，换前缀让老访客缓存的 api.swanlab.cn 失效。
      storage: { prefix: "--swanlab-oa" },
      // 接口示例提供 cURL / JavaScript / Python / Go 四种语言
      codeSamples: {
        defaultLang: "curl",
        generator: createCodeSampleGenerator([restApiSpec, restApiSpecEn]),
        availableLanguages: [
          {
            lang: "curl",
            label: "cURL",
            target: "shell",
            client: "curl",
            highlighter: "bash",
            icon: "curl",
          },
          {
            lang: "javascript",
            label: "JavaScript",
            target: "js",
            client: "fetch",
            highlighter: "javascript",
            icon: ".js",
          },
          {
            lang: "python",
            label: "Python",
            target: "python",
            client: "requests",
            highlighter: "python",
            icon: ".py",
          },
          {
            lang: "go",
            label: "Go",
            target: "go",
            client: "native",
            highlighter: "go",
            icon: ".go",
          },
        ],
      },
      i18n: { locale: "zh" },
    });
    openapiTheme.enhanceApp(ctx);
  },
  setup() {
    const route = useRoute();
    const { lang } = useData();
    const initVersionSync = useVersionSync();
    const initLogoLinkFix = useLogoLinkFix();

    // vitepress-openapi：UI 文案跟随当前语言（zh / en）
    watch(
      () => lang.value,
      (value) => {
        useOpenapiTheme({
          i18n: { locale: value.toLowerCase().startsWith("zh") ? "zh" : "en" },
        });
      },
      { immediate: true },
    );

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
