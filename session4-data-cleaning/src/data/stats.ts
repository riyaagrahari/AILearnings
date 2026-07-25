import raw from "./stats.json";
import type { Stats, Stage } from "../types";

export const stats = raw as unknown as Stats;

const byId = new Map<string, Stage>(stats.stages.map((s) => [s.id, s]));

export function stage(id: string): Stage {
  const s = byId.get(id);
  if (!s) throw new Error(`unknown stage: ${id}`);
  return s;
}
