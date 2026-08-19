/**
 * Locale-aware navbar logo link.
 *
 * The VoidZero OSS header hardcodes the top-left logo anchor (and the mobile
 * menu logo) to `/`, so clicking the logo on /en pages would jump back to the
 * Chinese homepage. The theme exposes no prop for this (`themeConfig.logoLink`
 * is only read by VPNavBarTitle, which the OSS header does not use), so the
 * rendered anchors are patched in place instead: `/en/` under the en locale,
 * `/` otherwise. Same DOM-patching technique as version-sync.ts — a
 * MutationObserver keeps the patch alive when the mobile menu (re)mounts.
 */
const LOGO_IMG_SELECTOR = 'header img[alt="SwanLab"], #mobile-menu img[alt="SwanLab"]';

function localeHome(): string {
  return location.pathname.split("/")[1] === "en" ? "/en/" : "/";
}

function patchLogoLinks(): void {
  const home = localeHome();
  for (const img of document.querySelectorAll<HTMLImageElement>(LOGO_IMG_SELECTOR)) {
    const anchor = img.closest("a");
    if (anchor && anchor.getAttribute("href") !== home) {
      anchor.setAttribute("href", home);
    }
  }
}

/** Returns the init function to invoke on `onMounted`. */
export function useLogoLinkFix(): () => void {
  return function initLogoLinkFix(): void {
    patchLogoLinks();
    // The desktop anchor persists across SPA navigation (Vue never re-renders
    // the hardcoded href), but the mobile menu mounts fresh with href="/"
    // each time it opens — observe the header root to re-patch on that.
    const navRoot = document.querySelector("header")?.parentElement;
    if (navRoot) {
      new MutationObserver(patchLogoLinks).observe(navRoot, {
        subtree: true,
        childList: true,
      });
    }
  };
}
