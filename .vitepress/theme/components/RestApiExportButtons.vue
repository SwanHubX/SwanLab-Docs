<template>
  <!-- 接口页头部操作条：Ask AI 下拉 + 复制 OpenAPI JSON。
       spec 里找不到 operationId 时整条不渲染（见 locatedOperation）。 -->
  <div v-if="locatedOperation" class="rest-export-buttons">
    <div class="rest-export-buttons-inner">
      <!-- 下拉的开关、点击外部关闭、图标渲染都在子组件里，这里只传菜单项 -->
      <AskAiDropdown :items="menuItems" />
      <button type="button" class="copy-page" @click="copyAsOpenapi">
        <span v-html="icons[copied ? 'check' : 'copy']" class="icon"></span>
        <span class="label">
          {{ copied ? labels.copied : labels.copy }}
        </span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onUnmounted, ref } from "vue";
import { useData } from "vitepress";
import restApiSpec from "../../specs/rest.zh.json" with { type: "json" };
import restApiSpecEn from "../../specs/rest.en.json" with { type: "json" };
import AskAiDropdown from "./AskAiDropdown.vue";

/** spec 里任意一个 JSON 对象节点（operation / parameter / schema 等）。 */
type JsonObject = Record<string, unknown>;

/** OpenAPI 的 security 字段：一条要求里，scheme 名 → 需要的 scope 列表。 */
type SecurityRequirement = Record<string, string[]>;

/** 本项目用到的 OpenAPI 文档子集，只声明导出逻辑会读到的字段。 */
type OpenApiDocument = {
  openapi: string;
  info: JsonObject & { title?: string };
  servers?: { url?: string }[];
  tags?: { name?: string }[];
  security?: SecurityRequirement[];
  paths: Record<string, JsonObject>;
  components?: Record<string, Record<string, JsonObject>>;
};

/** 定位到的接口：路径、HTTP 方法，以及它所在的 pathItem 和 operation 本身。 */
type LocatedOperation = {
  path: string;
  method: string;
  pathItem: JsonObject;
  operation: JsonObject;
};

/** 下拉里的一个 AI 入口：图标名、展示名、把提示词拼到 URL 上的前缀。 */
type AiProvider = {
  icon: string;
  name: string;
  promptUrl: string;
};

/** 路由参数里的接口 ID，由 `[operationId].md` 传给组件。 */
const props = defineProps<{
  operationId: string;
}>();

/** OpenAPI 允许出现在 pathItem 下的 HTTP 方法，用于遍历查找 operationId。 */
const METHODS = ["get", "post", "put", "patch", "delete", "head", "options", "trace"];

/** 中英文各一份 spec，按当前页面语言选用。 */
const specs: Record<"zh" | "en", OpenApiDocument> = {
  zh: restApiSpec as unknown as OpenApiDocument,
  en: restApiSpecEn as unknown as OpenApiDocument,
};

/** 把 icons/*.svg 以原始字符串读入，模板里用 v-html 内联渲染。 */
const rawModules = import.meta.glob("../icons/*.svg", {
  eager: true,
  query: "?raw",
  import: "default",
});

/** 文件名（不含 .svg）→ SVG 源码，供模板按图标名取用。 */
const icons = Object.fromEntries(
  Object.entries(rawModules).map(([iconPath, content]) => {
    const name = iconPath.match(/\/([^/]+)\.svg$/)?.[1] ?? "";
    return [name, content];
  }),
);

// 与 CopyOrDownloadAsMarkdownButtons.vue 保持一致：同样的三个 AI 入口。
const aiProviders: AiProvider[] = [
  {
    icon: "chatgpt",
    name: "ChatGPT",
    promptUrl: "https://chatgpt.com/?hints=search&prompt=",
  },
  {
    icon: "claude",
    name: "Claude",
    promptUrl: "https://claude.ai/new?q=",
  },
  {
    icon: "kimi",
    name: "Kimi",
    promptUrl: "https://www.kimi.com/_prefill_chat?prefill_prompt=",
  },
];

const { lang } = useData();

/** 当前页面是否为中文站点，决定用哪份 spec 和哪套按钮文案。 */
const isZh = computed(() => lang.value.toLowerCase().startsWith("zh"));

/** 按钮与菜单文案：查看/复制/已复制。 */
const labels = computed(() =>
  isZh.value
    ? {
        view: "查看 OpenAPI JSON",
        copy: "复制 OpenAPI JSON",
        copied: "已复制",
      }
    : {
        view: "View as OpenAPI JSON",
        copy: "Copy as OpenAPI JSON",
        copied: "Copied",
      },
);

/** 当前语言对应的 spec。 */
const spec = computed(() => specs[isZh.value ? "zh" : "en"]);

/**
 * 按 operationId 在当前页面里定位接口。
 * 定位失败（比如 spec 与路由不同步）时组件整体不渲染，避免页头出现空按钮组。
 */
const locatedOperation = computed(() => findOperation(spec.value, props.operationId));

// 菜单项：查看 JSON + 三个 AI 入口（与 CopyOrDownloadAsMarkdownButtons.vue 同一组入口）。
// 下拉组件只负责开关与关闭时机，每个菜单项做什么由这里决定。
const menuItems = computed(() => [
  { icon: "braces", label: labels.value.view, onSelect: viewAsOpenapi },
  ...aiProviders.map((provider) => ({
    icon: provider.icon,
    label: `Open in ${provider.name}`,
    onSelect: () => openInAI(provider),
  })),
]);

/** 复制成功后短暂切到「已复制」状态。 */
const copied = ref(false);

/** 「已复制」提示的复位计时器，连续点击时先清再设。 */
let copiedTimer: ReturnType<typeof setTimeout> | undefined;

/** 「已复制」状态的停留时长。 */
const COPIED_RESET_DELAY = 2000;

/** 把 unknown 收窄成 JSON 对象；数组、null、原始值都返回 undefined。 */
function asObject(value: unknown): JsonObject | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : undefined;
}

/** 把 unknown 收窄成数组；不是数组时给空数组，省掉调用处的判空。 */
function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

/** 把 unknown 收窄成字符串；非字符串返回 undefined。 */
function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

/** 还原 JSON Pointer 的转义：~1 → /，~0 → ~（RFC 6901）。 */
function decodePointerSegment(segment: string) {
  return segment.replace(/~1/g, "/").replace(/~0/g, "~");
}

/**
 * 解析文档内引用，如 "#/components/schemas/Error"。
 * 只支持本地片段引用（#/ 开头）；外部文件引用返回 undefined，导出时按缺失处理。
 */
function resolveRef(document: OpenApiDocument, ref: string): unknown {
  if (!ref.startsWith("#/")) return undefined;

  return ref
    .slice(2)
    .split("/")
    .map(decodePointerSegment)
    .reduce<unknown>((node, key) => asObject(node)?.[key], document);
}

/**
 * 一层解引用：节点本身是 { $ref } 就返回它指向的对象，否则原样返回。
 * 只解一层——嵌套引用由 collectComponentRefs 递归收集，避免互相引用时死循环。
 */
function dereference(document: OpenApiDocument, value: unknown): JsonObject | undefined {
  const object = asObject(value);
  const ref = asString(object?.$ref);

  if (ref) {
    return asObject(resolveRef(document, ref));
  }

  return object;
}

/**
 * 在整个文档里按 operationId 找接口。
 * spec 的 paths 是 路径 → 方法 → operation 的两层结构，所以两层都遍历；
 * 返回 pathItem 是为了后面能拿到 path 级 parameters（它们会补进导出的接口里）。
 */
function findOperation(
  document: OpenApiDocument,
  operationId: string,
): LocatedOperation | undefined {
  for (const [operationPath, pathItem] of Object.entries(document.paths ?? {})) {
    for (const method of METHODS) {
      const operation = asObject(pathItem[method]);

      if (operation && asString(operation.operationId) === operationId) {
        return { path: operationPath, method, pathItem, operation };
      }
    }
  }

  return undefined;
}

/**
 * 合并 path 级与 operation 级的 parameters。
 * OpenAPI 规定同名的（in + name 都相同）以 operation 级为准，
 * 所以 path 级里被覆盖的项要先剔掉，否则导出后会出现重复参数。
 */
function getParameters(document: OpenApiDocument, located: LocatedOperation): JsonObject[] {
  const pathParameters = asArray(located.pathItem.parameters)
    .map((parameter) => dereference(document, parameter))
    .filter((parameter): parameter is JsonObject => Boolean(parameter));
  const operationParameters = asArray(located.operation.parameters)
    .map((parameter) => dereference(document, parameter))
    .filter((parameter): parameter is JsonObject => Boolean(parameter));

  const operationKeys = new Set(
    operationParameters.map((parameter) => `${asString(parameter.in)}:${asString(parameter.name)}`),
  );

  return [
    ...pathParameters.filter(
      (parameter) => !operationKeys.has(`${asString(parameter.in)}:${asString(parameter.name)}`),
    ),
    ...operationParameters,
  ];
}

/**
 * 取接口实际生效的 security：operation 级优先，没有才回落到文档级的全局配置。
 * 注意 operation 里写 `security: []` 表示「显式不需要认证」，会原样返回空数组。
 */
function getSecurity(
  document: OpenApiDocument,
  operation: JsonObject,
): SecurityRequirement[] | undefined {
  return (operation.security as SecurityRequirement[] | undefined) ?? document.security;
}

/**
 * 递归收集一段 JSON 里所有 "#/components/**" 引用（含被引用对象内部再引用的），
 * 用于判断导出时需要带上哪些 components。
 * refs 既作为结果也作为已访问集合，所以循环引用不会无限递归。
 */
function collectComponentRefs(
  document: OpenApiDocument,
  value: unknown,
  refs = new Set<string>(),
): Set<string> {
  if (Array.isArray(value)) {
    value.forEach((item) => collectComponentRefs(document, item, refs));
    return refs;
  }

  const object = asObject(value);
  if (!object) return refs;

  const ref = asString(object.$ref);
  if (ref?.startsWith("#/components/")) {
    if (refs.has(ref)) return refs;

    refs.add(ref);
    collectComponentRefs(document, resolveRef(document, ref), refs);
    return refs;
  }

  Object.values(object).forEach((item) => collectComponentRefs(document, item, refs));

  return refs;
}

/**
 * 把收集到的 "#/components/<group>/<name>" 还原成 components 对象。
 * 引用目标不存在时跳过，保证导出结果不会带悬空 $ref。
 * 接口需要认证时会附上全部 securitySchemes——导出文件拿去校验时不会因为缺 scheme 报错。
 */
function buildComponents(
  document: OpenApiDocument,
  refs: Set<string>,
  security: SecurityRequirement[] | undefined,
): Record<string, Record<string, unknown>> | undefined {
  const components: Record<string, Record<string, unknown>> = {};

  for (const ref of refs) {
    const [, group, name] = ref.slice(2).split("/").map(decodePointerSegment);

    if (!group || !name) continue;

    const value = document.components?.[group]?.[name];

    if (!value) continue;

    components[group] ??= {};
    components[group][name] = value;
  }

  if (security?.length && document.components?.securitySchemes) {
    components.securitySchemes = document.components.securitySchemes;
  }

  return Object.keys(components).length ? components : undefined;
}

/**
 * 单接口导出：保留 info / servers / tags，只放当前 path + method，
 * 并带上它引用到的 components，保证导出的 JSON 自身可解析。
 * 接口未定位到（比如 operationId 拼错）时返回 undefined，调用方据此跳过导出。
 */
function buildExportDocument(): JsonObject | undefined {
  const located = locatedOperation.value;
  const document = spec.value;

  if (!located) return undefined;

  const parameters = getParameters(document, located);
  const operation: JsonObject = { ...located.operation };

  if (parameters.length) {
    operation.parameters = parameters;
  }

  const security = getSecurity(document, located.operation);
  const summary = asString(operation.summary);

  const refs = collectComponentRefs(document, operation);
  const components = buildComponents(document, refs, security);
  const title = asString(document.info.title) ?? "API";

  const exported: JsonObject = {
    openapi: document.openapi,
    info: {
      ...document.info,
      title: summary ? `${title} — ${summary}` : title,
    },
    servers: document.servers,
    tags: document.tags?.filter((tag) => asArray(operation.tags).includes(tag.name)),
    paths: { [located.path]: { [located.method]: operation } },
  };

  if (security?.length) {
    exported.security = security;
  }

  if (components) {
    exported.components = components;
  }

  return exported;
}

/**
 * 导出文档 → 带缩进的 JSON 字符串。
 * 定位不到接口时返回空串，调用方用真值判断即可跳过复制/预览。
 */
function exportJson() {
  const exported = buildExportDocument();

  return exported ? JSON.stringify(exported, null, 2) : "";
}

/** 拼给 AI 的提示词：让模型去读当前接口页的 URL。 */
function buildPrompt(pageUrl: string) {
  return `Read from ${pageUrl} so I can ask questions about it.`;
}

/**
 * 复制按钮：把当前接口的 OpenAPI JSON 写进剪贴板，成功后把按钮文案切到「已复制」。
 * 计时器先清再设，连续点击时不会出现提示提前消失。
 */
async function copyAsOpenapi() {
  try {
    await navigator.clipboard.writeText(exportJson());

    copied.value = true;
    clearTimeout(copiedTimer);
    copiedTimer = setTimeout(() => {
      copied.value = false;
    }, COPIED_RESET_DELAY);
  } catch (error) {
    console.error("Error copying OpenAPI document:", error);
  }
}

/**
 * 「查看 OpenAPI JSON」菜单项：新标签页打开导出的 JSON，不落盘。
 * 用 blob URL 而不是 data URL，长文档不会被 URL 长度限制截断；
 * 60 秒后再回收，避免新标签页还没读就被撤销。
 */
function viewAsOpenapi() {
  const text = exportJson();
  if (!text) return;

  // blob 打开纯文本，浏览器会直接显示 JSON，与 View as Markdown 的行为一致。
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);

  window.open(url, "_blank");
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

/** 「Open in <AI>」菜单项：把当前页面 URL 包进提示词，跳到对应 AI 的新对话页。 */
function openInAI(provider: AiProvider) {
  const prompt = buildPrompt(window.location.href);

  window.open(provider.promptUrl + encodeURIComponent(prompt), "_blank");
}

onUnmounted(() => {
  clearTimeout(copiedTimer);
});
</script>

<style scoped>
/* 操作条本身：占满整行，底部一条细线把按钮区与接口正文分开。 */
.rest-export-buttons {
  width: 100%;
  display: flex;
  margin: 12px 0 20px;
  padding-bottom: 14px;
  border-bottom: 1px solid color-mix(in srgb, var(--vp-c-divider) 50%, transparent);
}

.rest-export-buttons-inner {
  display: flex;
  align-items: center;
  gap: 10px;
}

/* 复制按钮：与 AskAiDropdown 里的触发按钮保持同一套视觉（尺寸、描边、字号）。
   两边各写一份是为了让两个组件互不依赖。 */
.copy-page {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 30px;
  padding: 5px 10px;
  background: transparent;
  border: 1.5px solid var(--vp-c-divider);
  border-radius: 6px;
  color: var(--vp-c-text-1);
  font-size: 12px;
  font-weight: 600;
  line-height: 16px;
  white-space: nowrap;
  cursor: pointer;
}

/* 文案随「已复制」切换长度，禁止换行避免按钮宽度抖动。 */
.label {
  white-space: nowrap;
}

/* SVG 以 v-html 内联，颜色靠 currentColor 继承。 */
.icon {
  width: 16px;
  height: 16px;
  flex: 0 0 16px;
  color: var(--vp-c-text-2);
}

/* 系统开启「减少动态效果」时不做过渡，只保留 hover 的颜色变化。 */
@media (prefers-reduced-motion: no-preference) {
  .copy-page {
    transition:
      background-color 0.16s ease,
      border-color 0.16s ease,
      color 0.16s ease;
  }

  .copy-page:hover {
    background: color-mix(in srgb, var(--vp-c-brand-1) 8%, transparent);
    border-color: color-mix(in srgb, var(--vp-c-brand-1) 40%, transparent);
    color: var(--vp-c-brand-1);
  }
}
</style>
