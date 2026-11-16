import { Check, Lightbulb, Play, Square, Trophy, X } from "lucide-react";
import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { defaultParams, GenParamsForm } from "../components/GenParamsForm";
import { PromptDiffView, versionLabel } from "../components/PromptBits";
import { Card, Empty, ErrorText, Field, ImageFrame, Notice, Pill, StatusBadge } from "../components/ui";
import { WithReconstruction } from "../components/WithReconstruction";
import { useApi } from "../hooks/useApi";
import { PageHeader } from "../layouts/AppLayout";
import { AXIS_TEXT, GRID, SINGLE_SERIES, TOOLTIP_STYLE } from "../lib/chart";
import { fmtNum } from "../lib/format";
import { api } from "../services/api";
import type { GenParams, Iteration, Reconstruction, RefinementSession } from "../types/api";

export function RefinementPage() {
  return (
    <>
      <PageHeader stage="Stage 5" title="Iterative refinement" description="Rule-based proposals compare the reference analysis with an analysis of the generated image. You approve or reject each proposal; the loop keeps the best iteration and stops early when the optimisation metric plateaus." />
      <WithReconstruction>{(r, reload) => <RefineBody r={r} reload={reload} />}</WithReconstruction>
    </>
  );
}

function RefineBody({ r, reload }: { r: Reconstruction; reload: () => Promise<void> }) {
  const sessions = r.refinement_sessions;
  const [sid, setSid] = useState(sessions.at(-1)?.id ?? "");
  useEffect(() => { if (!sid && sessions.length) setSid(sessions.at(-1)!.id); }, [sessions, sid]);
  const s = sessions.find((x) => x.id === sid);
  return (
    <div className="grid gap-6 xl:grid-cols-[360px_1fr]">
      <div className="space-y-6">
        <StartForm r={r} onStarted={async (id) => { await reload(); setSid(id); }} />
        {sessions.length > 0 && (
          <Card title="Sessions" pad={false}>
            <ul className="divide-y divide-ink-800">
              {[...sessions].reverse().map((x) => (
                <li key={x.id}><button onClick={() => setSid(x.id)} className={`flex w-full items-center justify-between px-4 py-2.5 text-left text-sm ${x.id === sid ? "bg-accent-500/10" : "hover:bg-ink-800"}`}>
                  <span><span className="mono text-slate-200">{x.id.slice(0, 6)}</span> <span className="text-slate-400">· {x.optimization_metric} · {x.iterations.length - 1} iter</span></span>
                  <StatusBadge status={x.status} />
                </button></li>
              ))}
            </ul>
          </Card>
        )}
      </div>
      {s ? <SessionView r={r} s={s} reload={reload} /> : <Empty title="No refinement session">Start one from a validated structured prompt (condition B).</Empty>}
    </div>
  );
}

function StartForm({ r, onStarted }: { r: Reconstruction; onStarted: (id: string) => Promise<void> }) {
  const sys = useApi(() => api.system(), []);
  const candidates = r.prompt_versions.filter((p) => p.source !== "caption");
  const [pv, setPv] = useState(candidates.filter((p) => p.source === "validated").at(-1)?.id ?? candidates.at(-1)?.id ?? "");
  const [metric, setMetric] = useState("clip_b32");
  const [maxIt, setMaxIt] = useState(5);
  const [patience, setPatience] = useState(2);
  const [minDelta, setMinDelta] = useState(0.005);
  const [analyze, setAnalyze] = useState("qwen3vl_local");
  const [params, setParams] = useState<GenParams>(defaultParams());
  const [error, setError] = useState<string | null>(null);
  const evalSet = sys.data?.metrics.filter((m) => m.default_role === "evaluation").map((m) => m.key) ?? [];
  useEffect(() => { const m = sys.data?.models[1]; if (m) setParams(defaultParams(m)); }, [sys.data]);

  const start = async () => {
    setError(null);
    try {
      const s = await api.startRefinement(r.id, { base_prompt_version_id: pv, optimization_metric: metric, max_iterations: maxIt, patience, min_delta: minDelta, provider: "gpu_worker", params, analysis_provider: analyze || null });
      await onStarted(s.id);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  if (!candidates.length) return <Card title="New session"><Empty title="No structured prompt">Create one in Prompt Studio.</Empty></Card>;
  return (
    <Card title="New session">
      <div className="space-y-3">
        <Field label="Base prompt"><select className="input" value={pv} onChange={(e) => setPv(e.target.value)}>{candidates.map((p) => <option key={p.id} value={p.id}>{versionLabel(p)}</option>)}</select></Field>
        <Field label="Optimisation metric" hint="Kept separate from the independent evaluation set.">
          <select className="input" value={metric} onChange={(e) => setMetric(e.target.value)}>{sys.data?.metrics.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}</select>
        </Field>
        {evalSet.includes(metric) && <Notice tone="warn">This metric is also in the independent evaluation set; its evaluation numbers for this session will not be independent.</Notice>}
        <div className="grid grid-cols-3 gap-2">
          <Field label="Max iter"><input className="input font-mono" type="number" min={1} max={50} value={maxIt} onChange={(e) => setMaxIt(Number(e.target.value))} /></Field>
          <Field label="Patience"><input className="input font-mono" type="number" min={0} value={patience} onChange={(e) => setPatience(Number(e.target.value))} /></Field>
          <Field label="Min Δ"><input className="input font-mono" type="number" step={0.001} min={0} value={minDelta} onChange={(e) => setMinDelta(Number(e.target.value))} /></Field>
        </div>
        <Field label="Analyse each output with" hint="Gives the proposer something to compare. Off = structural rules only.">
          <select className="input" value={analyze} onChange={(e) => setAnalyze(e.target.value)}>
            <option value="">Off</option>{sys.data?.vlm_providers.map((p) => <option key={p.name} value={p.name} disabled={!p.available}>{p.name}</option>)}
          </select>
        </Field>
        <GenParamsForm models={sys.data?.models ?? []} value={params} onChange={setParams} showNegative={false} />
        <ErrorText error={error} />
        <button className="btn-primary w-full" disabled={!pv} onClick={start}><Play className="h-4 w-4" />Start (generates baseline)</button>
      </div>
    </Card>
  );
}

function SessionView({ r, s, reload }: { r: Reconstruction; s: RefinementSession; reload: () => Promise<void> }) {
  const [error, setError] = useState<string | null>(null);
  const [maxEdits, setMaxEdits] = useState(3);
  const act = async (fn: () => Promise<unknown>) => { setError(null); try { await fn(); await reload(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } };
  const scored = s.iterations.filter((i) => i.status === "scored");
  const best = s.iterations.find((i) => i.id === s.best_iteration_id);
  const baseline = s.iterations[0];
  const pending = s.iterations.some((i) => ["proposed", "generating", "scoring", "approved"].includes(i.status));
  const genById = new Map(r.generations.map((g) => [g.id, g]));
  const chart = scored.map((i) => ({ iteration: i.iteration, score: i.optimization_score }));
  const executed = s.iterations.filter((i) => i.iteration > 0 && !["proposed", "rejected"].includes(i.status)).length;

  return (
    <div className="space-y-6">
      <Card title={<span className="flex items-center gap-2">Session {s.id.slice(0, 6)} <StatusBadge status={s.status} /></span>}
        subtitle={`Optimising ${s.optimization_metric} · ${executed}/${s.max_iterations} iterations · patience ${s.patience} · min Δ ${s.min_delta}`}
        actions={<>
          {s.status === "active" && <>
            <select className="input w-auto py-1.5" value={maxEdits} onChange={(e) => setMaxEdits(Number(e.target.value))}>{[1, 2, 3, 5].map((n) => <option key={n} value={n}>{n} edit{n > 1 ? "s" : ""}/proposal</option>)}</select>
            <button className="btn-primary" disabled={pending || !best} onClick={() => act(() => api.propose(s.id, maxEdits))}><Lightbulb className="h-4 w-4" />Propose next</button>
            <button className="btn-danger" onClick={() => act(() => api.stopRefinement(s.id))}><Square className="h-4 w-4" />Stop</button>
          </>}
        </>}>
        {s.stop_reason && <Notice tone="info">{s.stop_reason}</Notice>}
        {s.generation_config.warnings?.map((w, i) => <Notice key={i} tone="warn">{w}</Notice>)}
        <ErrorText error={error} />
        {chart.length ? (
          <div className="mt-2 h-56">
            <ResponsiveContainer>
              <LineChart data={chart} margin={{ top: 12, right: 16, bottom: 0, left: -8 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="iteration" tick={{ fill: AXIS_TEXT, fontSize: 11 }} axisLine={{ stroke: GRID }} tickLine={false} label={{ value: "iteration", position: "insideBottomRight", fill: AXIS_TEXT, fontSize: 11, offset: -2 }} />
                <YAxis domain={["auto", "auto"]} tick={{ fill: AXIS_TEXT, fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => fmtNum(v, 3)} />
                <Tooltip {...TOOLTIP_STYLE} cursor={{ stroke: "#475569" }} formatter={(v: number) => [fmtNum(v), s.optimization_metric]} labelFormatter={(l) => `Iteration ${l}`} />
                <Line type="monotone" dataKey="score" stroke={SINGLE_SERIES} strokeWidth={2} dot={{ r: 4, fill: SINGLE_SERIES, stroke: "#0b1019", strokeWidth: 2 }} isAnimationActive={false} />
                {best?.optimization_score != null && <ReferenceDot x={best.iteration} y={best.optimization_score} r={7} fill="none" stroke="#fab219" strokeWidth={2} label={{ value: "best", fill: "#e2e8f0", fontSize: 11, position: "top" }} />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : <p className="py-6 text-center text-sm text-slate-500">Waiting for the baseline to be generated and scored by the GPU worker…</p>}
      </Card>

      <div className="space-y-3">
        {[...s.iterations].reverse().map((it) => (
          <IterationCard key={it.id} it={it} best={it.id === s.best_iteration_id} imageUrl={it.generation_id ? genById.get(it.generation_id)?.output_image?.url : undefined}
            onApprove={(idx) => act(() => api.approve(it.id, idx))} onReject={(reason) => act(() => api.reject(it.id, reason))} />
        ))}
      </div>

      {best && baseline && best.prompt_version_id && baseline.prompt_version_id && best.id !== baseline.id && (
        <Card title="Baseline → best prompt"><PromptDiffView a={baseline.prompt_version_id} b={best.prompt_version_id} /></Card>
      )}
    </div>
  );
}

function IterationCard({ it, best, imageUrl, onApprove, onReject }: {
  it: Iteration; best: boolean; imageUrl?: string; onApprove: (idx: number[] | null) => void; onReject: (reason: string) => void;
}) {
  const edits = it.proposal?.edits ?? [];
  const [sel, setSel] = useState<number[]>(edits.map((_, i) => i));
  const [reason, setReason] = useState("");
  return (
    <div className={`card flex flex-col gap-4 p-4 md:flex-row ${best ? "ring-1 ring-amber-400/50" : ""}`}>
      <div className="w-full shrink-0 md:w-40">
        <ImageFrame src={imageUrl} />
      </div>
      <div className="min-w-0 flex-1 space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-semibold text-slate-100">{it.iteration === 0 ? "Baseline" : `Iteration ${it.iteration}`}</span>
          <StatusBadge status={it.status} />
          {best && <Pill tone="amber"><Trophy className="mr-1 h-3 w-3" />best</Pill>}
          <span className="ml-auto font-mono text-lg text-slate-100">{it.status === "scored" ? fmtNum(it.optimization_score) : "—"}</span>
        </div>
        {it.proposal?.rationale && <p className="text-sm text-slate-400">{it.proposal.rationale}{it.proposal.base_iteration !== undefined && ` Based on iteration ${it.proposal.base_iteration}.`}</p>}
        {edits.length > 0 && (
          <ul className="space-y-1.5">
            {edits.map((e, i) => (
              <li key={i} className="flex gap-2.5 rounded-md border border-ink-700 bg-ink-850 px-3 py-2 text-sm">
                {it.status === "proposed" && <input type="checkbox" className="mt-1" checked={sel.includes(i)} onChange={(ev) => setSel(ev.target.checked ? [...sel, i] : sel.filter((x) => x !== i))} />}
                <div>
                  <p className="text-slate-200"><Pill tone="accent">{e.op}</Pill> <span className="text-slate-400">{e.field}</span> → <span className="font-medium">{e.value}</span>{e.old_value ? <span className="text-slate-500"> (was “{e.old_value}”)</span> : null}</p>
                  <p className="mt-0.5 text-xs text-slate-500">{e.reason}</p>
                </div>
              </li>
            ))}
          </ul>
        )}
        {it.proposal?.rejection_reason && <p className="text-xs text-slate-500">Rejected: {it.proposal.rejection_reason}</p>}
        {it.status === "proposed" && (
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <button className="btn-primary" disabled={!sel.length} onClick={() => onApprove(sel.length === edits.length ? null : sel)}><Check className="h-4 w-4" />Approve {sel.length !== edits.length ? `${sel.length} edit(s)` : ""}</button>
            <input className="input w-56 py-1.5" placeholder="Reason (optional)" value={reason} onChange={(e) => setReason(e.target.value)} />
            <button className="btn-secondary" onClick={() => onReject(reason)}><X className="h-4 w-4" />Reject</button>
          </div>
        )}
      </div>
    </div>
  );
}
