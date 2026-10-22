<script setup lang="ts">
import { useRoute } from "vitepress";
import restApiSpec from "../../../.vitepress/specs/rest.en.json" with { type: "json" };

const route = useRoute();
const operationId = route.data.params?.operationId as string;
</script>

<div class="swanlab-openapi">

<ClientOnly>
  <OAOperation :operationId="operationId" :spec="restApiSpec">
    <template #header="{ operation }">
      <div class="swanlab-openapi-header">
        <OAHeading level="h1" class="scroll-m-(--vp-nav-height)">
          {{ operation.summary }}
        </OAHeading>
        <RestApiExportButtons :operation-id="operationId" />
      </div>
    </template>
  </OAOperation>
</ClientOnly>

</div>

<style>
/* 接口页用整页宽度，不要右边那一列——那里只有一个空的目录。
   本来想在 md 顶部写 aside: false，但这里不生效：
   构建时 llms 插件会在正文前插一行隐藏文字，frontmatter 就不在文件开头了，VitePress 认不出来。
   所以改用 CSS 实现，:has() 保证只作用于 /api/rest/ 下的接口页。 */
.VPDoc:has(.vp-doc[class*="_api_rest_"]) .aside {
  display: none;
}

.VPDoc:has(.vp-doc[class*="_api_rest_"]) .content-container {
  max-width: none !important;
}

/* 自定义 header：标题 + 导出按钮，与内置 header 的纵向排列保持一致 */
.swanlab-openapi-header {
  display: flex;
  flex-direction: column;
}
</style>
