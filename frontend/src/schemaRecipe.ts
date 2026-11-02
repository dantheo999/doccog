import type { Recipe } from "./types";
import type { Collection, ExtractionSchema } from "./workbench/types";

/** Whether a recipe was built for this exact agreed schema version. */
export function recipeMatchesSchema(recipe: Recipe | null, schema: ExtractionSchema | null): boolean {
  return !!recipe && !!schema && recipe.schema_id === schema.id && recipe.schema_version === schema.version;
}

/** Why the recipe does not fit the agreed schema, or null when it does (or there is nothing to compare). */
export function recipeSchemaMismatch(recipe: Recipe | null, schema: ExtractionSchema | null): string | null {
  if (!recipe || !schema || recipeMatchesSchema(recipe, schema)) return null;
  if (recipe.schema_id === schema.id && recipe.schema_version)
    return `The active recipe was built for v${recipe.schema_version} of “${schema.name}”; the agreed schema is now v${schema.version}.`;
  return `The active recipe was not built from the agreed schema “${schema.name}” v${schema.version}, so its classes may not match the schema fields.`;
}

/** Exemplars for a schema's recipe: the ticked documents, else the documents the schema was suggested from
 * (matched by name in the current document set). At most five. */
export function schemaExemplarIds(schema: ExtractionSchema, collection: Collection | null, ticked: string[] = []): string[] {
  if (ticked.length) return ticked.slice(0, 5);
  if (!collection) return [];
  const names = new Set(schema.exemplars);
  return collection.documents.filter(item => names.has(item.name)).map(item => item.id).slice(0, 5);
}
