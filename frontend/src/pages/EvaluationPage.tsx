import { Calculator } from "lucide-react";
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, Empty, ErrorText, ImageFrame, MetricTile, Notice, Pill, StatusBadge } from "../components/ui";
import { WithReconstruction } from "../components/WithReconstruction";
import { useApi } from "../hooks/useApi";
import { PageHeader } from "../layouts/AppLayout";
import { AXIS_TEXT, GRID, SINGLE_SERIES, TOOLTIP_STYLE } from "../lib/chart";
import { CONDITION_LABEL, fmtNum } from "../lib/format";
import { api } from "../services/api";
import type { Generation, Reconstruction } from "../types/api";

export function EvaluationPage() {
  return (
    <>
      <PageHeader stage="Stage 4" title="Evaluation" description="Each metric is computed and reported independently. A metric that has not run shows “—” with its status — values are never estimated or back-filled." />
      <WithReconstruction>{(r, reload) => <EvalBody r={r} reload={reload} />}</WithReconstruction>
    </>
  );
}

function EvalBody({ r, reload }: { r: Reconstruction; reload: () => Promise<void> }) {
  const sys = useApi(() => api.system(), []);
  const done = r.generations.filter((g) => g.status === "completed");
  const [gid, setGid] = useState(done.at(-1)?.id ?? "");
  const [extra, setExtra] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { if (!gid && done.length) setGid(done.at(-1)!.id); }, [done, gid]);
  const g = done.find((x) => x.id === gid);

  if (!done.length) return <Empty title="No completed generations">Generate an image in Stage 3; default evaluation metrics are queued automatically when it completes.</Empty>;

  const request = async () => {
    setError(null);
    try { await api.evaluate(gid, extra); setExtra([]); await reload(); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const evalMetrics = g?.metrics.filter((m) => m.role === "evaluation") ?? [];
  const optMetrics = g?.metrics.filter((m) => m.role === "optimization") ?? [];
  const metricKeys = sys.data?.metrics.map((m) => m.key) ?? [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        {done.map((x) => (
          <button key={x.id} onClick={() => setGid(x.id)} className={`flex items-center gap-2 rounded-lg border p-1.5 pr-3 text-left text-xs transition ${x.id === gid ? "border-accent-400 bg-accent-500/10" : "border-ink-700 bg-ink-850 hover:border-ink-500"}`}>
            <img src={x.output_image?.url} alt="" className="h-10 w-10 rounded object-cover" />
            <span><span className="block text-slate-200">v{x.prompt_version} · {x.model_key}</span><span className="text-slate-500">seed {x.params.seed}{x.condition ? ` · ${x.condition[0]}` : ""}</span></span>
          </button>
        ))}
      </div>

      {g && (
        <div className="grid gap-6 xl:grid-cols-[1fr_1.1fr]">
          <div className="grid grid-cols-2 gap-4">
            <ImageFrame src={r.image.url} label="Reference" sub={`${r.image.width}×${r.image.height}`} />
            <ImageFrame src={g.output_image?.url} label={<span className="flex items-center gap-2">Generated <StatusBadge status={g.status} /></span>}
              sub={<>{g.condition && <Pill tone="accent">{CONDITION_LABEL[g.condition]}</Pill>} {g.model_id}</>} />
            <p className="mono col-span-2 rounded-lg border border-ink-700 bg-ink-950/70 p-3 text-slate-300">{g.positive}</p>
          </div>
          <div className="space-y-5">
            <div>
              <h3 className="label mb-2">Independent evaluation metrics</h3>
              {evalMetrics.length ? <div className="grid gap-3 sm:grid-cols-2">{evalMetrics.map((m) => <MetricTile key={m.id} m={m} />)}</div>
                : <p className="text-sm text-slate-500">None requested.</p>}
            </div>
            {optMetrics.length > 0 && (
              <div>
                <h3 className="label mb-2">Refinement objective (not an independent evaluation)</h3>
                <div className="grid gap-3 sm:grid-cols-2">{optMetrics.map((m) => <MetricTile key={m.id} m={m} />)}</div>
              </div>
            )}
            <Card title="Request more metrics">
              <div className="flex flex-wrap gap-2">
                {metricKeys.filter((k) => !evalMetrics.some((m) => m.metric === k && m.status !== "failed")).map((k) => (
                  <label key={k} className="flex items-center gap-2 rounded-md border border-ink-600 bg-ink-850 px-2.5 py-1.5 text-sm">
                    <input type="checkbox" checked={extra.includes(k)} onChange={(e) => setExtra(e.target.checked ? [...extra, k] : extra.filter((x) => x !== k))} />
                    {sys.data?.metrics.find((m) => m.key === k)?.label}
                  </label>
                ))}
              </div>
              <ErrorText error={error} />
              <button className="btn-secondary mt-3" disabled={!extra.length} onClick={request}><Calculator className="h-4 w-4" />Queue metrics</button>
            </Card>
          </div>
        </div>
      )}

      <MetricCharts gens={done} />
      <Notice tone="info" title="Reading the metrics">CLIP and DINOv2 are cosine similarities of image embeddings (semantic / structural content). SSIM compares aligned pixels and penalises any change of layout, so it is expected to be low for prompt-based reconstruction. LPIPS is a distance: lower means more similar.</Notice>
    </div>
  );
}

/** Small multiples: one chart per metric (different scales never share an axis). */
function MetricCharts({ gens }: { gens: Generation[] }) {
  const byMetric = new Map<string, { label: string; hib: boolean; rows: { name: string; value: number; detail: string }[] }>();
  for (const g of gens) {
    for (const m of g.metrics) {
      if (m.role !== "evaluation" || m.status !== "completed" || m.value === null) continue;
      const e = byMetric.get(m.metric) ?? { label: m.label, hib: m.higher_is_better, rows: [] };
      e.rows.push({ name: `v${g.prompt_version}·${g.params.seed}${g.condition ? `·${g.condition[0]}` : ""}`, value: m.value, detail: `${g.model_key}, seed ${g.params.seed}` });
      byMetric.set(m.metric, e);
    }
  }
  if (!byMetric.size) return <Empty title="No completed metric values yet">Charts appear once metrics have genuinely been computed.</Empty>;
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {[...byMetric.entries()].map(([key, e]) => (
        <Card key={key} title={e.label} subtitle={`${e.hib ? "Higher" : "Lower"} = closer to reference · n = ${e.rows.length}`}>
          <div className="h-56">
            <ResponsiveContainer>
              <BarChart data={e.rows} margin={{ top: 8, right: 8, bottom: 0, left: -8 }} barCategoryGap={6}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="name" tick={{ fill: AXIS_TEXT, fontSize: 11 }} axisLine={{ stroke: GRID }} tickLine={false} />
                <YAxis tick={{ fill: AXIS_TEXT, fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => fmtNum(v, 2)} />
                <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [fmtNum(v), e.label]} />
                <Bar dataKey="value" fill={SINGLE_SERIES} radius={[4, 4, 0, 0]} maxBarSize={36} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      ))}
    </div>
  );
}
