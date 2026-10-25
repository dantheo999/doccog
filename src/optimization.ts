import { z } from "zod";
import { recipeSchema } from "./recipe";
import type { OptimizationProposal, Recipe } from "./types";

export const optimizationProposalSchema = z.object({
  base_recipe_id: z.string().trim().min(1).max(100),
  base_recipe_version: z.number().int().min(1),
  candidate: recipeSchema,
  reasons: z.array(z.string().trim().min(1)).min(1).max(10),
}).strict().superRefine((proposal, context) => {
  if (proposal.candidate.id !== proposal.base_recipe_id
    || proposal.candidate.version !== proposal.base_recipe_version + 1) {
    context.addIssue({ code: "custom", path: ["candidate"],
      message: "Optimization candidate must keep the base recipe ID and use exactly the next version." });
  }
});

export function parseOptimizationProposal(value: unknown): OptimizationProposal {
  const result = optimizationProposalSchema.safeParse(value);
  if (!result.success) throw new Error(`Invalid optimization proposal: ${result.error.issues.map(issue =>
    `${issue.path.join(".")}: ${issue.message}`).join("\n")}`);
  return result.data;
}

export function assertOptimizationBase(proposal: OptimizationProposal, recipe: Recipe) {
  if (proposal.base_recipe_id !== recipe.id || proposal.base_recipe_version !== recipe.version
    || proposal.candidate.id !== recipe.id || proposal.candidate.version !== recipe.version + 1
    || proposal.candidate.goal !== recipe.goal) {
    throw new Error("Optimization does not match the active recipe and goal. Extract again before optimizing.");
  }
  if (proposal.candidate.examples.length !== recipe.examples.length
    || proposal.candidate.examples.some((example, index) => {
      const original = recipe.examples[index];
      return example.text !== original.text
        || JSON.stringify(example.source ?? null) !== JSON.stringify(original.source ?? null)
        || (!original.source?.verified
          && JSON.stringify(example.extractions) !== JSON.stringify(original.extractions));
    })) {
    throw new Error("Optimization must retain every original exemplar's text, provenance and order, "
      + "and keep unverified examples unchanged. Discard this proposal and run Optimize again.");
  }
}
