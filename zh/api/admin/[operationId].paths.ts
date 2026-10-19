import { usePaths } from "vitepress-openapi";
import spec from "../../../.vitepress/specs/admin.json" with { type: "json" };

export default {
  paths() {
    return usePaths({ spec })
      .getPathsByVerbs()
      .map(({ operationId, summary }) => ({
        params: {
          operationId,
          pageTitle: `${summary} - 管理后台开放 API`,
        },
      }));
  },
};
