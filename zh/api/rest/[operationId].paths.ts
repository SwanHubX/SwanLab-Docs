import { usePaths } from "vitepress-openapi";
import spec from "../../../.vitepress/specs/rest.zh.json" with { type: "json" };

export default {
  paths() {
    return usePaths({ spec })
      .getPathsByVerbs()
      .map(({ operationId, summary }) => ({
        params: {
          operationId,
          pageTitle: `${summary} - REST API`,
        },
      }));
  },
};
