/**
 * JSON Schema to zod, for re-advertising a bridged device's tools
 * (src/plugins/hardware.ts). A device describes its tool arguments in JSON
 * Schema (pydantic on a Python server, zod-to-json-schema on a TypeScript one);
 * the MCP SDK registers tools from zod. This covers the subset tool arguments
 * use: primitives (numbers with their bounds), enums and consts, arrays,
 * objects and maps, nullable unions (`anyOf` with `null`, or a `type` list),
 * local `$ref`s into `$defs` / `definitions`, descriptions and defaults.
 *
 * Anything outside that subset becomes `z.unknown()`: the argument still
 * reaches the device as sent, and the device validates it either way. A
 * converted schema may be looser than the device's, never stricter.
 */
import { z, type ZodRawShape, type ZodTypeAny } from 'zod';

type JsonObject = { readonly [key: string]: unknown };

interface Walk {
  readonly rootDefs: JsonObject;
  readonly refHops: number;
}

const MAX_REF_HOPS = 16;

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function objectAt(node: JsonObject, key: string): JsonObject {
  const value = node[key];
  return isObject(value) ? value : {};
}

function isPrimitive(value: unknown): value is string | number | boolean | null {
  return value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean';
}

function convert(node: unknown, walk: Walk): ZodTypeAny {
  if (!isObject(node)) return z.unknown();
  const schema = convertNode(node, walk);
  return typeof node.description === 'string' ? schema.describe(node.description) : schema;
}

function convertNode(node: JsonObject, walk: Walk): ZodTypeAny {
  if (typeof node.$ref === 'string') {
    if (walk.refHops >= MAX_REF_HOPS) return z.unknown();
    return convert(resolveRef(node.$ref, walk.rootDefs), { ...walk, refHops: walk.refHops + 1 });
  }
  if (Array.isArray(node.enum)) return fromValues(node.enum);
  if ('const' in node) return fromValues([node.const]);
  const variants = Array.isArray(node.anyOf) ? node.anyOf : Array.isArray(node.oneOf) ? node.oneOf : undefined;
  if (variants) return fromVariants(variants, walk);
  if (Array.isArray(node.allOf) && node.allOf.length === 1) return convert(node.allOf[0], walk);
  if (Array.isArray(node.type)) {
    return fromVariants(node.type.map((type: unknown) => ({ ...node, type, description: undefined })), walk);
  }
  return typeof node.type === 'string' ? fromType(node.type, node, walk) : z.unknown();
}

/** `#/$defs/Name` or `#/definitions/Name`; any other reference resolves to nothing (accept anything). */
function resolveRef(ref: string, defs: JsonObject): unknown {
  const name = /^#\/(?:\$defs|definitions)\/([^/]+)$/.exec(ref)?.[1];
  return name === undefined ? undefined : defs[name.replace(/~1/g, '/').replace(/~0/g, '~')];
}

function fromValues(values: readonly unknown[]): ZodTypeAny {
  const literals = values.filter(isPrimitive).map((value) => z.literal(value));
  const [first, second, ...rest] = literals;
  if (first === undefined || literals.length !== values.length) return z.unknown();
  return second === undefined ? first : z.union([first, second, ...rest]);
}

function fromVariants(variants: readonly unknown[], walk: Walk): ZodTypeAny {
  const isNull = (variant: unknown): boolean => isObject(variant) && variant.type === 'null';
  const [first, second, ...rest] = variants.filter((variant) => !isNull(variant)).map((variant) => convert(variant, walk));
  if (first === undefined) return z.null();
  const union = second === undefined ? first : z.union([first, second, ...rest]);
  return variants.some(isNull) ? union.nullable() : union;
}

function fromType(type: string, node: JsonObject, walk: Walk): ZodTypeAny {
  switch (type) {
    case 'string':
      return z.string();
    case 'number':
      return withBounds(z.number(), node);
    case 'integer':
      return withBounds(z.number().int(), node);
    case 'boolean':
      return z.boolean();
    case 'null':
      return z.null();
    case 'array':
      return z.array(isObject(node.items) ? convert(node.items, walk) : z.unknown());
    case 'object':
      return fromObject(node, walk);
    default:
      return z.unknown();
  }
}

function withBounds(schema: z.ZodNumber, node: JsonObject): z.ZodNumber {
  const { minimum, maximum, exclusiveMinimum, exclusiveMaximum } = node;
  const atLeast = typeof minimum === 'number' ? schema.min(minimum) : schema;
  const atMost = typeof maximum === 'number' ? atLeast.max(maximum) : atLeast;
  const above = typeof exclusiveMinimum === 'number' ? atMost.gt(exclusiveMinimum) : atMost;
  return typeof exclusiveMaximum === 'number' ? above.lt(exclusiveMaximum) : above;
}

function fromObject(node: JsonObject, walk: Walk): ZodTypeAny {
  if (isObject(node.properties)) {
    const object = z.object(shapeOf(node, walk));
    return node.additionalProperties === false ? object.strict() : object.passthrough();
  }
  return z.record(isObject(node.additionalProperties) ? convert(node.additionalProperties, walk) : z.unknown());
}

function shapeOf(node: JsonObject, walk: Walk): ZodRawShape {
  const required = new Set(Array.isArray(node.required) ? node.required : []);
  return Object.fromEntries(
    Object.entries(objectAt(node, 'properties')).map(([key, property]) => [
      key,
      argument(convert(property, walk), property, required.has(key)),
    ]),
  );
}

/** Required as is; optional with the default the device declared when that value fits, so clients see it. */
function argument(schema: ZodTypeAny, property: unknown, required: boolean): ZodTypeAny {
  if (required) return schema;
  if (isObject(property) && 'default' in property && schema.safeParse(property.default).success) {
    return schema.default(property.default);
  }
  return schema.optional();
}

/**
 * The zod raw shape for a tool's JSON Schema `inputSchema` (an object schema).
 * @param schema - the device's inputSchema, as listed by its tools/list.
 * @returns one zod schema per declared argument, ready for `ctx.mcp.tool`.
 */
export function jsonSchemaToShape(schema: unknown): ZodRawShape {
  if (!isObject(schema)) return {};
  const rootDefs = { ...objectAt(schema, 'definitions'), ...objectAt(schema, '$defs') };
  return shapeOf(schema, { rootDefs, refHops: 0 });
}
