<script setup lang="ts">
import { computed, useSlots } from "vue";
import { useData } from "vitepress";
import VPDefaultLayout from "@voidzero-dev/vitepress-theme/src/components/vitepress-default/Layout.vue";

// Always render through the VoidZero theme's forked VitePress default layout.
// The upstream VPLayout routes `layout: home` pages to the VoidZero marketing
// layout (which would drop the hero/features homepage), so we bypass that
// router and keep every page on the forked docs skeleton instead.
const { frontmatter, site } = useData();
const slots = useSlots();

const variant = computed(() => site.value.themeConfig?.variant || "voidzero");
</script>

<template>
  <div class="docs-layout" :data-theme="frontmatter.theme" :data-variant="variant">
    <VPDefaultLayout>
      <!-- Forward all slots to the default layout -->
      <template v-for="(_, name) in slots" #[name]="slotData">
        <slot :name="name" v-bind="slotData || {}" />
      </template>
    </VPDefaultLayout>
  </div>
</template>
