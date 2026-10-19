import { usePaths } from "vitepress-openapi";
import spec from "../../../.vitepress/specs/admin.en.json" with { type: "json" };

export default {
  paths() {
    return usePaths({ spec })
      .getPathsByVerbs()
      .map(({ operationId, summary }) => ({
        params: {
          operationId,
          pageTitle: `${summary} - Admin API`,
        },
      }));
  },
};
