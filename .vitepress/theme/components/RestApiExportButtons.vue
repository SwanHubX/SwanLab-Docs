<template>
  <div v-if="locatedOperation" class="rest-export-buttons">
    <div class="rest-export-buttons-inner">
      <div class="dropdown-container" ref="dropdownContainer">
        <button class="open-page" :aria-expanded="isOpen" @click.stop="toggleDropdown">
          <span v-html="icons['sparkle']" class="icon"></span>
          <span class="label">Ask AI</span>
          <span v-html="icons['chevron']" class="icon chevron" :class="{ open: isOpen }"></span>
        </button>

        <div v-if="isRendered" ref="dropdownMenu" class="dropdown-menu" :class="{ open: isOpen }">
          <button class="dropdown-item" @click="viewAsOpenapi">
            <span v-html="icons['braces']" class="icon"></span>
            {{ labels.view }}
            <span v-html="icons['external']" class="icon external"></span>
          </button>

          <button
            v-for="provider in aiProviders"
            :key="provider.name"
            class="dropdown-item"
            @click="openInAI(provider)"
          >
            <span v-html="icons[provider.icon]" class="icon"></span>
            Open in {{ provider.name }}
            <span v-html="icons['external']" class="icon external"></span>
          </button>
        </div>
      </div>
      <button class="copy-page" @click="copyAsOpenapi">
        <span v-html="icons[copied ? 'check' : 'copy']" class="icon"></span>
        <span class="label">
          {{ copied ? labels.copied : labels.copy }}
        </span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { useData } from "vitepress";
import restApiSpec from "../../specs/rest.zh.json" with { type: "json" };
import restApiSpecEn from "../../specs/rest.en.json" with { type: "json" };

type JsonObject = Record<string, unknown>;

type SecurityRequirement = Record<string, string[]>;

type OpenApiDocument = {
  openapi: string;
  info: JsonObject & { title?: string };
  servers?: { url?: string }[];
  tags?: { name?: string }[];
  security?: SecurityRequirement[];
  paths: Record<string, JsonObject>;
  components?: {
    schemas?: Record<string, JsonObject>;
    securitySchemes?: Record<string, JsonObject>;
  };
};

type LocatedOperation = {
  path: string;
  method: string;
  pathItem: JsonObject;
  operation: JsonObject;
};

type AiProvider = {
  icon: string;
  name: string;
  promptUrl: string;
  buildPrompt: (pageUrl: string) => string;
};

const props = defineProps<{
  operationId: string;
}>();

const METHODS = ["get", "post", "put", "patch", "delete", "head", "options", "trace"];

const specs: Record<"zh" | "en", OpenApiDocument> = {
  zh: restApiSpec as unknown as OpenApiDocument,
  en: restApiSpecEn as unknown as OpenApiDocument,
};

const rawModules = import.meta.glob("../icons/*.svg", {
  eager: true,
  query: "?raw",
  import: "default",
});

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
    buildPrompt: (pageUrl) => `Read from ${pageUrl} so I can ask questions about it.`,
  },
  {
    icon: "claude",
    name: "Claude",
    promptUrl: "https://claude.ai/new?q=",
    buildPrompt: (pageUrl) => `Read from ${pageUrl} so I can ask questions about it.`,
  },
  {
    icon: "kimi",
    name: "Kimi",
    promptUrl: "https://www.kimi.com/_prefill_chat?prefill_prompt=",
    buildPrompt: (pageUrl) => `Read from ${pageUrl} so I can ask questions about it.`,
  },
];

const { lang } = useData();

const isZh = computed(() => lang.value.toLowerCase().startsWith("zh"));

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

const spec = computed(() => specs[isZh.value ? "zh" : "en"]);

const locatedOperation = computed(() => findOperation(spec.value, props.operationId));

const isOpen = ref(false);
const copied = ref(false);
const dropdownContainer = ref<HTMLElement>();
const dropdownMenu = ref<HTMLElement>();
const isRendered = ref(false);

let copiedTimer: ReturnType<typeof setTimeout> | undefined;

const animationDuration = 2000;

function asObject(value: unknown): JsonObject | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : undefined;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function decodePointerSegment(segment: string) {
  return segment.replace(/~1/g, "/").replace(/~0/g, "~");
}

function resolveRef(document: OpenApiDocument, ref: string): unknown {
  if (!ref.startsWith("#/")) return undefined;

  return ref
    .slice(2)
    .split("/")
    .map(decodePointerSegment)
    .reduce<unknown>((node, key) => asObject(node)?.[key], document);
}

function dereference(document: OpenApiDocument, value: unknown): JsonObject | undefined {
  const object = asObject(value);
  const ref = asString(object?.$ref);

  if (ref) {
    return asObject(resolveRef(document, ref));
  }

  return object;
}

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

function getSecurity(
  document: OpenApiDocument,
  operation: JsonObject,
): SecurityRequirement[] | undefined {
  return (operation.security as SecurityRequirement[] | undefined) ?? document.security;
}

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

function buildComponents(
  document: OpenApiDocument,
  refs: Set<string>,
  security: SecurityRequirement[] | undefined,
): JsonObject | undefined {
  const components: Record<string, Record<string, unknown>> = {};

  for (const ref of refs) {
    const [, group, name] = ref.slice(2).split("/").map(decodePointerSegment);

    if (!group || !name) continue;

    const componentGroup = document.components?.[group as "schemas" | "securitySchemes"];
    const value = componentGroup?.[name];

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
  const pathItem: JsonObject = { [located.method]: operation };
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
    paths: { [located.path]: pathItem },
  };

  if (security?.length) {
    exported.security = security;
  }

  if (components) {
    exported.components = components;
  }

  return exported;
}

function exportJson() {
  const exported = buildExportDocument();

  return exported ? JSON.stringify(exported, null, 2) : "";
}

function getPageURL() {
  if (typeof window === "undefined") return "";

  return window.location.href;
}

async function copyAsOpenapi() {
  try {
    await navigator.clipboard.writeText(exportJson());

    copied.value = true;
    copiedTimer = setTimeout(() => {
      copied.value = false;
    }, animationDuration);
  } catch (error) {
    console.error("Error copying OpenAPI document:", error);
  }

  isOpen.value = false;
}

function viewAsOpenapi() {
  const text = exportJson();
  if (!text) return;

  // blob 打开纯文本，浏览器会直接显示 JSON，与 View as Markdown 的行为一致。
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);

  window.open(url, "_blank");
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
  isOpen.value = false;
}

function openInAI(provider: AiProvider) {
  const prompt = provider.buildPrompt(getPageURL());

  window.open(provider.promptUrl + encodeURIComponent(prompt), "_blank");
  isOpen.value = false;
}

function toggleDropdown() {
  if (isOpen.value) {
    isOpen.value = false;

    const el = dropdownMenu.value;
    if (!el) {
      return;
    }

    const onEnd = () => {
      isRendered.value = false;
      el.removeEventListener("transitionend", onEnd);
    };

    el.addEventListener("transitionend", onEnd);
  } else {
    isRendered.value = true;
    requestAnimationFrame(() => {
      isOpen.value = true;
    });
  }
}

function handleClickOutside(event: MouseEvent) {
  if (dropdownContainer.value && !dropdownContainer.value.contains(event.target as Node)) {
    isOpen.value = false;
  }
}

onMounted(() => document.addEventListener("click", handleClickOutside));
onUnmounted(() => {
  document.removeEventListener("click", handleClickOutside);
  clearTimeout(copiedTimer);
});
</script>

<style scoped>
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
  position: relative;
}

.dropdown-container {
  position: relative;
}

.copy-page,
.open-page {
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

.label {
  white-space: nowrap;
}

.dropdown-menu {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  width: 222px;
  padding: 4px;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  z-index: 30;
  box-shadow:
    0 10px 24px rgba(0, 0, 0, 0.12),
    0 2px 8px rgba(0, 0, 0, 0.08);
  opacity: 0;
  transform: translateY(-4px) scale(0.98);
  pointer-events: none;
}

.dropdown-menu.open {
  opacity: 1;
  transform: translateY(0) scale(1);
  pointer-events: auto;
}

.dropdown-item {
  position: relative;
  width: 100%;
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 36px;
  padding: 8px;
  background: transparent;
  border: none;
  border-radius: 8px;
  color: var(--vp-c-text-1);
  font-size: 14px;
  line-height: 20px;
  cursor: pointer;
  text-align: left;
}

.dropdown-item .icon.external {
  margin-left: auto;
  opacity: 0.6;
}

.icon {
  width: 16px;
  height: 16px;
  flex: 0 0 16px;
  color: var(--vp-c-text-2);
}

.dropdown-item .icon {
  width: 18px;
  height: 18px;
  flex-basis: 18px;
}

.chevron.open {
  transform: rotate(180deg);
}

.dropdown-item:hover .icon.external {
  opacity: 1;
  transform: translateX(2px);
}

@media (prefers-reduced-motion: no-preference) {
  .dropdown-menu {
    transition:
      opacity 0.18s cubic-bezier(0.4, 0, 0.2, 1),
      transform 0.18s cubic-bezier(0.4, 0, 0.2, 1);
    transform-origin: top;
  }

  .copy-page,
  .open-page,
  .dropdown-item,
  .dropdown-item .icon.external {
    transition:
      background-color 0.16s ease,
      border-color 0.16s ease,
      color 0.16s ease,
      transform 0.16s ease,
      opacity 0.16s ease;
  }

  .copy-page:hover,
  .open-page:hover {
    background: color-mix(in srgb, var(--vp-c-brand-1) 8%, transparent);
    border-color: color-mix(in srgb, var(--vp-c-brand-1) 40%, transparent);
    color: var(--vp-c-brand-1);
  }

  .dropdown-item:hover {
    background: var(--vp-c-bg-soft);
  }

  .chevron {
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }
}
</style>
