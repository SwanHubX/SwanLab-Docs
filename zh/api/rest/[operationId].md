<script setup lang="ts">
import { useRoute } from "vitepress";

const route = useRoute();
const operationId = route.data.params?.operationId as string;
</script>

<div class="swanlab-openapi">

<ClientOnly>
  <OAOperation :operationId="operationId">
    <template #header="{ operation }">
      <div class="swanlab-openapi-header">
        <OAHeading level="h1" class="scroll-m-(--vp-nav-height)">
          {{ operation.summary }}
        </OAHeading>
        <RestApiExportButtons :operation-id="operationId" />
      </div>
    </template>
    <template #code-samples="{ operationId: id }">
      <OAHeading level="h2">代码示例</OAHeading>
      <OACodeSamples :operation-id="id" />
    </template>
  </OAOperation>
</ClientOnly>

</div>

<style>
/* 接口页没有 Markdown 标题，右侧目录永远是空的，所以只隐藏这一列。
   正文宽度不再覆盖，和普通文档一样保持居中，避免刷新时的宽度跳变。
   :has(.swanlab-openapi) 只匹配接口页，不影响 /api/rest/ 概述页。 */
.VPDoc:has(.swanlab-openapi) .aside {
  display: none;
}

/* 自定义 header：标题 + 导出按钮，与内置 header 的纵向排列保持一致 */
.swanlab-openapi-header {
  display: flex;
  flex-direction: column;
}
</style>
