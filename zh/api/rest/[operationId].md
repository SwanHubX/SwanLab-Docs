---
aside: false
outline: false
---

<script setup lang="ts">
import { useRoute } from "vitepress";

const route = useRoute();
const operationId = route.data.params?.operationId as string;
</script>

<div class="swanlab-openapi">

<ClientOnly>
  <OAOperation :operationId="operationId" />
</ClientOnly>

</div>
