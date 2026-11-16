/** Chart tokens. Categorical slots 1–4 assigned in fixed order along the display order A → D → B → C and
 *  validated with the dataviz validator against the #0b1019 surface (dark mode, the only theme):
 *  lightness band, chroma, CVD ΔE ≥ 8.4 (protan, yellow↔aqua), normal-vision ΔE ≥ 19.8, contrast ≥ 3:1 — all PASS.
 *  Refinement views other than "final" are not charted (no fifth/sixth colour passes adjacency checks). */
export const CONDITION_COLOR: Record<string, string> = {
  A_caption: "#3987e5",       // slot 1 · blue
  D_detailed: "#d95926",      // slot 2 · orange
  B_structured: "#199e70",    // slot 3 · aqua
  C_refined: "#c98500",       // slot 4 · yellow
  C_refined_final: "#c98500", // same entity as C_refined
};
export const SINGLE_SERIES = "#3987e5";
export const GRID = "#1d2738";
export const AXIS_TEXT = "#94a3b8";
export const TOOLTIP_STYLE = {
  contentStyle: { background: "#0f1622", border: "1px solid #2a3649", borderRadius: 8, fontSize: 12, color: "#e2e8f0" },
  labelStyle: { color: "#cbd5e1" },
  itemStyle: { color: "#e2e8f0" },
  cursor: { fill: "rgba(148,163,184,0.08)" },
};
