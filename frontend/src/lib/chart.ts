/** Chart tokens. Categorical slots validated with the dataviz validator against the #0b1019 surface
 *  (dark mode): lightness, chroma, CVD ΔE ≥ 9.4, normal-vision ΔE ≥ 26.5, contrast ≥ 3:1 — all PASS. */
export const CONDITION_COLOR: Record<string, string> = {
  A_caption: "#3987e5",    // slot 1 · blue
  B_structured: "#d95926", // slot 2 · orange
  C_refined: "#199e70",    // slot 3 · aqua
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
