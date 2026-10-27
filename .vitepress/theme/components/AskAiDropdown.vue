<template>
  <div ref="container" class="dropdown-container">
    <button type="button" class="open-page" :aria-expanded="isOpen" @click.stop="isOpen = !isOpen">
      <span v-html="icons['sparkle']" class="icon"></span>
      <span class="label">Ask AI</span>
      <span v-html="icons['chevron']" class="icon chevron" :class="{ open: isOpen }"></span>
    </button>

    <!-- 菜单常驻 DOM，只切 .open：关闭过渡由 CSS 完成，不需要等 transitionend 再卸载。
         inert 让关闭状态下的菜单项不参与 Tab 聚焦。 -->
    <div class="dropdown-menu" :class="{ open: isOpen }" :inert="!isOpen || undefined">
      <button
        v-for="item in items"
        :key="item.label"
        type="button"
        class="dropdown-item"
        @click="select(item)"
      >
        <span v-html="icons[item.icon]" class="icon"></span>
        {{ item.label }}
        <span v-html="icons['external']" class="icon external"></span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";

defineProps<{
  items: { icon: string; label: string; onSelect: () => void }[];
}>();

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

const isOpen = ref(false);
const container = ref<HTMLElement>();

function select(item: { onSelect: () => void }) {
  item.onSelect();
  isOpen.value = false;
}

function handleClickOutside(event: MouseEvent) {
  if (container.value && !container.value.contains(event.target as Node)) {
    isOpen.value = false;
  }
}

onMounted(() => document.addEventListener("click", handleClickOutside));
onUnmounted(() => document.removeEventListener("click", handleClickOutside));
</script>

<style scoped>
.dropdown-container {
  position: relative;
}

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
