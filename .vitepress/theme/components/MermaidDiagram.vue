<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, useId, watch } from "vue";

const props = defineProps<{
  graph: string;
}>();

const container = ref<HTMLElement>();
const componentId = useId().replaceAll(":", "");
let colorSchemeObserver: MutationObserver | undefined;
let renderSequence = 0;
let diagramSequence = 0;

async function renderDiagram() {
  const host = container.value;

  if (!host) return;

  const currentRender = ++renderSequence;
  const { default: mermaid } = await import("mermaid");

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: document.documentElement.classList.contains("dark") ? "dark" : "default",
  });

  try {
    const id = `mermaid-${componentId}-${diagramSequence++}`;
    const { svg, bindFunctions } = await mermaid.render(id, decodeURIComponent(props.graph));

    if (currentRender !== renderSequence || !container.value) return;

    container.value.innerHTML = svg;
    bindFunctions?.(container.value);
  } catch (error) {
    if (currentRender !== renderSequence || !container.value) return;

    console.error("Failed to render Mermaid diagram", error);
    container.value.textContent = "Mermaid diagram failed to render.";
  }
}

onMounted(() => {
  void renderDiagram();

  colorSchemeObserver = new MutationObserver(() => void renderDiagram());
  colorSchemeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
});

watch(
  () => props.graph,
  () => void nextTick(renderDiagram),
);

onBeforeUnmount(() => {
  renderSequence++;
  colorSchemeObserver?.disconnect();
});
</script>

<template>
  <div ref="container" class="mermaid-diagram" />
</template>

<style scoped>
.mermaid-diagram {
  margin: 16px 0;
  overflow-x: auto;
  text-align: center;
}

.mermaid-diagram :deep(svg) {
  max-width: 100%;
  height: auto;
}
</style>
