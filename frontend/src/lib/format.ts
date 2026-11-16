export const fmtNum = (v: number | null | undefined, digits = 4) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const fmtTime = (iso: string | null | undefined) => {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
};

export const fmtDuration = (s: number | null | undefined) =>
  s === null || s === undefined ? "—" : s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(2)} s`;

export const shortId = (id: string | null | undefined) => (id ? id.slice(0, 8) : "—");

export const CONDITION_LABEL: Record<string, string> = {
  A_caption: "A · Caption only",
  D_detailed: "D · Detailed prose prompt",
  B_structured: "B · Structured prompt",
  C_refined: "C · Structured + refinement",
};

/** Analysis groups. Refinement is split into views; "best" is selected post hoc and therefore biased. */
export const GROUP_LABEL: Record<string, string> = {
  A_caption: "A · Caption", D_detailed: "D · Detailed prose", B_structured: "B · Structured",
  C_iter0: "C · iteration 0 (baseline)", C_refined_final: "C · Refined (final iteration)",
  C_refined_best: "C · Refined (best — selected, biased)", unlabelled: "Unlabelled",
};

/** Charted groups (one validated colour each). C_iter0 and C_refined_best appear in tables only. */
export const CHART_GROUPS = ["A_caption", "D_detailed", "B_structured", "C_refined_final"] as const;

export const FIELD_LABEL: Record<string, string> = {
  subject: "Subject", secondary_objects: "Secondary objects", spatial_relationships: "Spatial relationships",
  composition: "Composition", camera_angle: "Camera angle", lighting: "Lighting", colors: "Colours",
  materials: "Materials", textures: "Textures", background: "Background", art_style: "Art style",
  extra_positive: "Extra positive terms", extra_negative: "Extra negative terms",
};

export const emptyAttributes = () => ({
  subject: "", secondary_objects: [], spatial_relationships: [], composition: "", camera_angle: "", lighting: "",
  colors: [], materials: [], textures: [], background: "", art_style: "", extra_positive: [], extra_negative: [],
});
