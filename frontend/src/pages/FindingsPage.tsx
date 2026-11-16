import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, Empty, Notice, Spinner } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { PageHeader } from "../layouts/AppLayout";
import { AXIS_TEXT, CONDITION_COLOR, GRID, TOOLTIP_STYLE } from "../lib/chart";
import { CONDITION_LABEL, fmtNum } from "../lib/format";
import { api } from "../services/api";

const ORDER = ["A_caption", "B_structured", "C_refined", "unlabelled"];

export function FindingsPage() {
  const stats = useApi(() => api.stats(), []);
  const sys = useApi(() => api.system(), []);
  const data = stats.data?.by_metric_and_condition ?? {};
  const metricInfo = new Map(sys.data?.metrics.map((m) => [m.key, m]));
  const keys = Object.keys(data);

  return (
    <>
      <PageHeader title="Research findings" description="Aggregated from completed, independent evaluation metrics only. Each metric gets its own chart because their scales differ. Missing values are excluded, never imputed — check n before drawing conclusions." />
      <div className="mb-6"><Notice tone="info" title="Interpretation guard-rails">Means over few images are anecdotal. Compare conditions only within the same experiment, model and seed set. Refinement (C) is selected by its optimisation metric, so prefer the other metrics when judging whether C genuinely improves on B.</Notice></div>
      {!stats.data ? <Spinner /> : keys.length === 0 ? (
        <Empty title="No genuine results recorded yet">Findings will appear after real generations have been evaluated on the GPU worker. This page deliberately shows nothing rather than placeholder numbers.</Empty>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          {keys.map((k) => {
            const info = metricInfo.get(k);
            const rows = ORDER.filter((c) => data[k][c]).map((c) => ({ condition: c, label: CONDITION_LABEL[c] ?? "Unlabelled", ...data[k][c] }));
            return (
              <Card key={k} title={info?.label ?? k} subtitle={`${info?.higher_is_better === false ? "Lower" : "Higher"} = closer to reference · mean of completed values`}>
                <div className="h-60">
                  <ResponsiveContainer>
                    <BarChart data={rows} margin={{ top: 20, right: 8, bottom: 0, left: -8 }} barCategoryGap="30%">
                      <CartesianGrid stroke={GRID} vertical={false} />
                      <XAxis dataKey="label" tick={{ fill: AXIS_TEXT, fontSize: 11 }} axisLine={{ stroke: GRID }} tickLine={false} />
                      <YAxis tick={{ fill: AXIS_TEXT, fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => fmtNum(v, 2)} />
                      <Tooltip {...TOOLTIP_STYLE} formatter={(v: number, _n, p) => [`${fmtNum(v)} (n=${p.payload.n}, range ${fmtNum(p.payload.min, 3)}–${fmtNum(p.payload.max, 3)})`, "mean"]} />
                      <Bar dataKey="mean" radius={[4, 4, 0, 0]} maxBarSize={56}>
                        {rows.map((r) => <Cell key={r.condition} fill={CONDITION_COLOR[r.condition] ?? "#64748b"} />)}
                        <LabelList dataKey="n" position="top" formatter={(n: number) => `n=${n}`} fill="#cbd5e1" fontSize={11} />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <table className="mt-3 w-full text-sm">
                  <thead className="text-left text-xs text-slate-400"><tr><th className="py-1 font-medium">Condition</th><th className="font-medium">n</th><th className="font-medium">mean</th><th className="font-medium">min</th><th className="font-medium">max</th></tr></thead>
                  <tbody className="font-mono text-slate-300">{rows.map((r) => (
                    <tr key={r.condition} className="border-t border-ink-800"><td className="py-1 font-sans"><span className="mr-2 inline-block h-2.5 w-2.5 rounded-sm" style={{ background: CONDITION_COLOR[r.condition] ?? "#64748b" }} />{r.label}</td><td>{r.n}</td><td>{fmtNum(r.mean)}</td><td>{fmtNum(r.min)}</td><td>{fmtNum(r.max)}</td></tr>
                  ))}</tbody>
                </table>
              </Card>
            );
          })}
        </div>
      )}
    </>
  );
}
