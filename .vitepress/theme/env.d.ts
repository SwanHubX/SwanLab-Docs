declare module "*.vue" {
  import type { DefineComponent } from "vue";
  const component: DefineComponent<{}, {}, any>;
  export default component;
}

declare module "virtual:group-icons.css";

/** Build-time baked SwanLab version, injected via `vite.define` in config.mts. */
declare const __SWANLAB_VERSION__: string;
