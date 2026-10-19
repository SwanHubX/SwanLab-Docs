---
aside: false
outline: false
---

<script setup lang="ts">
import { useRoute } from "vitepress";
import adminApiSpec from "../../../.vitepress/specs/admin.en.json" with { type: "json" };

const route = useRoute();
const operationId = route.data.params?.operationId as string;
</script>

<div class="swanlab-openapi">

<ClientOnly>
  <OAOperation :operationId="operationId" :spec="adminApiSpec" />
</ClientOnly>

</div>
