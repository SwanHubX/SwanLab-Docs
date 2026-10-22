import type { OARequest } from "vitepress-openapi/client";
import { generateCodeSample, OARequest as SwanLabRequest } from "vitepress-openapi/client";

/**
 * vitepress-openapi derives the Authorization header from the security scheme,
 * but an `apiKey` scheme without a token has no value to use, so it falls back
 * to the header name itself — every secured sample came out as
 * `Authorization: Authorization`. Map each scheme to the prefix the service
 * actually expects (see SwanLab-Auth `SplitAuth`).
 */
const AUTH_HEADER_SAMPLES: Record<string, string> = {
  basicAuth: "Basic <base64(app_id:app_secret)>",
  bearerAuth: "Bearer <access_token>",
  apiKeyAuth: "ApiKey <root-api-key>",
};

function withAuthHeader(request: OARequest): OARequest {
  const schemes = request.authorizations;
  const scheme = Array.isArray(schemes) ? schemes[0] : schemes;
  const value = scheme?.label ? AUTH_HEADER_SAMPLES[scheme.label] : undefined;

  if (!value) return request;

  return new SwanLabRequest({
    ...request,
    headers: { ...request.headers, authorization: value },
  });
}

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
    const authorized = withAuthHeader(request);

    if (authorized.body !== undefined) {
      return generateCodeSample(langConfig, authorized);
    }

    const example = bodyExamples.find(
      (item) => item.method === authorized.method && item.pathPattern.test(authorized.path ?? ""),
    );

    if (!example) {
      return generateCodeSample(langConfig, authorized);
    }

    return generateCodeSample(
      langConfig,
      new SwanLabRequest({ ...authorized, body: example.value }),
    );
  };
}
