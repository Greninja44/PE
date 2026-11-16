import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, Empty, Notice, Pill, Spinner } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { PageHeader } from "../layouts/AppLayout";
import { AXIS_TEXT, CONDITION_COLOR, GRID, TOOLTIP_STYLE } from "../lib/chart";
import { CHART_GROUPS, fmtNum, GROUP_LABEL } from "../lib/format";
import { api } from "../services/api";
import type { Aggregate } from "../types/api";

const METRICS = ["clip_l14", "dinov2_base", "ssim", "lpips_alex"];

/** Every statement on this page carries one of these evidence types. */
const EVIDENCE: Record<string, { label: string; tone: "emerald" | "cyan" | "accent" | "amber" | "slate" | "rose"; text: string }> = {
  measured: { label: "Measured", tone: "emerald", text: "Computed by this installation from recorded runs." },
  human: { label: "Human evaluation", tone: "cyan", text: "Ratings entered by people (Compare & Rate), kept apart from metrics." },
  suggestion: { label: "Model / rule suggestion", tone: "accent", text: "Proposed by the VLM or the rule-based refiner; not evidence of improvement." },
  hypothesis: { label: "Hypothesis", tone: "amber", text: "A claim under test, not a result." },
  future: { label: "Future work", tone: "slate", text: "Planned, not implemented or not yet run." },
  reported: { label: "Reported prior finding", tone: "rose", text: "Stated by the researcher; raw data not in this repository, so it cannot be re-computed here." },
};

function Ev({ k }: { k: keyof typeof EVIDENCE }) {
  return <Pill tone={EVIDENCE[k].tone}>{EVIDENCE[k].label}</Pill>;
}

export function FindingsPage() {
  const stats = useApi(() => api.stats(), []);
  const agg = stats.data?.comparison;

  return (
    <>
      <PageHeader title="Research findings" description="Aggregated from completed, independent evaluation metrics only. Each metric gets its own chart because their scales differ. Missing values are excluded, never imputed — check n before drawing conclusions." />

      <div className="mb-6 flex flex-wrap gap-2" aria-label="Evidence types">
        {Object.entries(EVIDENCE).map(([k, v]) => <span key={k} title={v.text}><Pill tone={v.tone}>{v.label}</Pill></span>)}
      </div>

      <div className="mb-6 grid gap-6 lg:grid-cols-2">
        <Card title={<span className="flex items-center gap-2">Research question <Ev k="hypothesis" /></span>}>
          <p className="text-sm leading-relaxed text-slate-300">Do structured reconstructed prompts (B), detailed prose prompts (D) and iterative refinement (C) reproduce a
            reference image more faithfully than an ordinary caption (A), when model, seed, steps, guidance and resolution are held fixed?</p>
        </Card>
        <Card title={<span className="flex items-center gap-2">Earlier pilot, n = 12 images <Ev k="reported" /></span>}>
          <ul className="list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-slate-300">
            <li>Structured prompts did <b>not</b> consistently outperform plain captions.</li>
            <li>Apparent refinement improvements were potentially affected by <b>selection bias</b> (choosing the best iteration after the fact).</li>
          </ul>
          <p className="mt-3 text-xs text-slate-500">The pilot's images, prompts and scores are not in this repository, so they are not plotted or recomputed here. To include them, import the original files unchanged.</p>
        </Card>
      </div>

      {!stats.data ? <Spinner /> : !agg || Object.keys(agg.groups).length === 0 ? (
        <Empty title="No genuine results recorded yet">Findings will appear after real generations have been evaluated by the inference worker. This page deliberately shows nothing rather than placeholder numbers.</Empty>
      ) : (
        <Measured agg={agg} />
      )}

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="Interpretation guard-rails">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-300">
            <li>Means over few images are anecdotal; report n and the paired win/loss counts, not only means.</li>
            <li>Compare conditions only within the same model, seed and settings (the paired table does this automatically).</li>
            <li>Refinement optimises one metric (default CLIP B/32); judge it with the <i>other</i> metrics. “Best” iterations are selected post hoc and biased upward — use “final”.</li>
            <li>Refinement edit proposals are <Ev k="suggestion" /> — they become evidence only after generation and independent evaluation.</li>
          </ul>
        </Card>
        <Card title={<span className="flex items-center gap-2">Future work <Ev k="future" /></span>}>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-300">
            <li>Repeat the comparison on a larger, pre-registered image set with several seeds per image.</li>
            <li>Statistical testing of paired differences (e.g. Wilcoxon signed-rank) once n is adequate.</li>
            <li>Inter-rater agreement for human ratings (several blind raters per image).</li>
            <li>Larger VLMs / SDXL on hardware with more memory; quantised VLM inference on Apple Silicon.</li>
          </ul>
        </Card>
      </div>
    </>
  );
}

function Measured({ agg }: { agg: Aggregate }) {
  const charted = CHART_GROUPS.filter((g) => agg.groups[g]);
  const allGroups = Object.keys(agg.groups);
  return (
    <div className="space-y-6">
      <Notice tone="info" title={`Measured · ${agg.n_completed_generations} completed generations across ${agg.n_reconstructions} reference image(s)`}>
        {agg.notes.join(" ")}
      </Notice>

      <div className="grid gap-6 lg:grid-cols-2">
        {METRICS.filter((k) => charted.some((g) => agg.groups[g].metrics[k])).map((k) => {
          const info = agg.metrics[k];
          const rows = charted.filter((g) => agg.groups[g].metrics[k]).map((g) => {
            const s = agg.groups[g].metrics[k];
            return { group: g, label: GROUP_LABEL[g], mean: s.mean, n: s.n, sd: s.sd ?? 0, min: s.min, max: s.max };
          });
          return (
            <Card key={k} title={<span className="flex items-center gap-2">{info.label} <Ev k="measured" /></span>}
              subtitle={`${info.higher_is_better ? "↑ Higher" : "↓ Lower"} = closer to reference · mean ± 1 SD of completed values`}>
              <div className="h-60" role="img" aria-label={`${info.label} by condition; table view below`}>
                <ResponsiveContainer>
                  <BarChart data={rows} margin={{ top: 22, right: 8, bottom: 0, left: -8 }} barCategoryGap="30%">
                    <CartesianGrid stroke={GRID} vertical={false} />
                    <XAxis dataKey="label" tick={{ fill: AXIS_TEXT, fontSize: 11 }} axisLine={{ stroke: GRID }} tickLine={false} interval={0} />
                    <YAxis tick={{ fill: AXIS_TEXT, fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => fmtNum(v, 2)} />
                    <Tooltip {...TOOLTIP_STYLE} formatter={(v: number, _n, p) => [`${fmtNum(v)} (n=${p.payload.n}, SD ${fmtNum(p.payload.sd, 3)}, range ${fmtNum(p.payload.min, 3)}–${fmtNum(p.payload.max, 3)})`, "mean"]} />
                    <Bar dataKey="mean" radius={[4, 4, 0, 0]} maxBarSize={56}>
                      {rows.map((r) => <Cell key={r.group} fill={CONDITION_COLOR[r.group] ?? "#64748b"} />)}
                      <ErrorBar dataKey="sd" width={6} stroke="#94a3b8" strokeWidth={1} />
                      <LabelList dataKey="n" position="top" offset={14} formatter={(n: number) => `n=${n}`} fill="#cbd5e1" fontSize={11} />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          );
        })}
      </div>

      <Card pad={false} title={<span className="flex items-center gap-2">All groups (table view) <Ev k="measured" /> <Ev k="human" /></span>}
        subtitle="Includes refinement iteration 0 and the post-hoc best iteration, which are not charted. Groups overlap — do not add their n.">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[900px] text-sm">
            <thead className="text-left text-xs text-slate-400"><tr className="border-b border-ink-700">
              <th className="px-4 py-2.5 font-medium">Group</th>
              {METRICS.map((k) => <th key={k} className="px-3 py-2.5 text-right font-medium">{agg.metrics[k]?.label.replace(" similarity", "")} {agg.metrics[k]?.higher_is_better ? "↑" : "↓"}</th>)}
              <th className="px-3 py-2.5 text-right font-medium">Gen. time</th>
              <th className="px-3 py-2.5 text-right font-medium">≈CLIP tokens</th>
              <th className="px-3 py-2.5 text-right font-medium">Iterations run</th>
              <th className="px-3 py-2.5 text-right font-medium">Human 1–5</th>
            </tr></thead>
            <tbody className="font-mono text-slate-300">
              {allGroups.map((g) => {
                const G = agg.groups[g];
                const cell = (s?: { mean: number; n: number } | null, d = 3) => s ? <>{fmtNum(s.mean, d)}<span className="ml-1 text-[11px] text-slate-500">n={s.n}</span></> : <span className="text-slate-600">—</span>;
                return (
                  <tr key={g} className="border-b border-ink-800 last:border-0">
                    <td className="px-4 py-2 font-sans">
                      <span className="mr-2 inline-block h-2.5 w-2.5 rounded-sm" style={{ background: CONDITION_COLOR[g] ?? "transparent", border: CONDITION_COLOR[g] ? undefined : "1px solid #64748b" }} />
                      {GROUP_LABEL[g] ?? g}{G.selection_biased && <span className="ml-2"><Pill tone="amber">selection-biased</Pill></span>}
                    </td>
                    {METRICS.map((k) => <td key={k} className="px-3 py-2 text-right">{cell(G.metrics[k])}</td>)}
                    <td className="px-3 py-2 text-right">{G.other.generation_s ? `${fmtNum(G.other.generation_s.mean, 1)} s` : "—"}</td>
                    <td className="px-3 py-2 text-right">{G.other.prompt_clip_tokens_approx ? fmtNum(G.other.prompt_clip_tokens_approx.mean, 0) : "—"}</td>
                    <td className="px-3 py-2 text-right">{G.other.iterations_executed ? fmtNum(G.other.iterations_executed.mean, 1) : "—"}</td>
                    <td className="px-3 py-2 text-right">{cell(G.human_similarity, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card pad={false} title={<span className="flex items-center gap-2">Paired comparisons <Ev k="measured" /></span>}
        subtitle="Same reference image and identical generation settings only. Counts say how often the second condition was closer to the reference, in each metric's own direction.">
        {agg.paired.length === 0 ? <div className="p-5"><Empty title="No complete pairs yet">Run several conditions with identical settings on the same reference image.</Empty></div> : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-sm">
              <thead className="text-left text-xs text-slate-400"><tr className="border-b border-ink-700">
                <th className="px-4 py-2.5 font-medium">Comparison (1st → 2nd)</th><th className="px-3 py-2.5 font-medium">Metric</th>
                <th className="px-3 py-2.5 text-right font-medium">Pairs</th><th className="px-3 py-2.5 text-right font-medium">2nd closer</th>
                <th className="px-3 py-2.5 text-right font-medium">2nd further</th><th className="px-3 py-2.5 text-right font-medium">Ties</th>
                <th className="px-3 py-2.5 text-right font-medium">Mean Δ (2nd − 1st)</th>
              </tr></thead>
              <tbody className="text-slate-300">
                {agg.paired.map((p) => (
                  <tr key={`${p.a}-${p.b}-${p.metric}`} className="border-b border-ink-800 last:border-0">
                    <td className="px-4 py-2">{GROUP_LABEL[p.a]} → {GROUP_LABEL[p.b]}</td>
                    <td className="px-3 py-2">{agg.metrics[p.metric]?.label} {p.higher_is_better ? "↑" : "↓"}</td>
                    <td className="px-3 py-2 text-right font-mono">{p.n_pairs}</td>
                    <td className="px-3 py-2 text-right font-mono">{p.b_better}</td>
                    <td className="px-3 py-2 text-right font-mono">{p.b_worse}</td>
                    <td className="px-3 py-2 text-right font-mono">{p.ties}</td>
                    <td className="px-3 py-2 text-right font-mono">{fmtNum(p.mean_diff_b_minus_a, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
