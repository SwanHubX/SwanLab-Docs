import type { OARequest } from "vitepress-openapi/client";
import { generateCodeSample, OARequest as SwanLabRequest } from "vitepress-openapi/client";

/**
 * vitepress-openapi 0.2.5 builds its sample request body with
 * `content[type].examples[0].value`, but `examples` is an OpenAPI map
 * (`{ JSON: { value } }`), so `[0]` is always undefined and POST/PATCH samples
 * lose their body. Collect request body examples from the specs and inject them
 * at generation time, keeping the workaround outside node_modules.
 */
type OpenApiDocument = {
  paths?: Record<
    string,
    Record<
      string,
      {
        requestBody?: {
          content?: Record<
            string,
            {
              example?: unknown;
              examples?: Record<string, { value?: unknown }>;
            }
          >;
        };
      }
    >
  >;
};

type BodyExample = {
  method: string;
  pathPattern: RegExp;
  value: unknown;
};

const BODY_METHODS = new Set(["post", "put", "patch", "delete"]);

function toPathPattern(path: string): RegExp {
  const pattern = path
    .split("/")
    .map((segment) =>
      /^\{[^}]+\}$/.test(segment) ? "[^/]+" : segment.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
    )
    .join("/");

  return new RegExp(`^${pattern}$`);
}

function collectBodyExamples(specs: OpenApiDocument[]): BodyExample[] {
  const examples: BodyExample[] = [];

  for (const spec of specs) {
    for (const [path, pathItem] of Object.entries(spec.paths ?? {})) {
      for (const method of BODY_METHODS) {
        const media = pathItem[method]?.requestBody?.content?.["application/json"];
        if (!media) continue;

        const value =
          media.example ??
          Object.values(media.examples ?? {}).find((entry) => entry?.value !== undefined)?.value;
        if (value === undefined) continue;

        examples.push({
          method: method.toUpperCase(),
          pathPattern: toPathPattern(path),
          value,
        });
      }
    }
  }

  return examples;
}

export function createCodeSampleGenerator(specs: OpenApiDocument[]) {
  const bodyExamples = collectBodyExamples(specs);

  return (langConfig: Parameters<typeof generateCodeSample>[0], request: OARequest) => {
    if (request.body !== undefined) {
      return generateCodeSample(langConfig, request);
    }

    const example = bodyExamples.find(
      (item) => item.method === request.method && item.pathPattern.test(request.path ?? ""),
    );

    if (!example) {
      return generateCodeSample(langConfig, request);
    }

    return generateCodeSample(langConfig, new SwanLabRequest({ ...request, body: example.value }));
  };
}
