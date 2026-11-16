import clsx from "clsx";
import { ArrowDown, ArrowUp, EyeOff, Shuffle, Star, UserCheck } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Card, Empty, ErrorText, Field, ImageFrame, Notice, Pill, Spinner } from "../components/ui";
import { WithReconstruction } from "../components/WithReconstruction";
import { useApi } from "../hooks/useApi";
import { PageHeader } from "../layouts/AppLayout";
import { CONDITION_COLOR } from "../lib/chart";
import { fmtDuration, fmtNum, GROUP_LABEL } from "../lib/format";
import { api } from "../services/api";
import type { ComparisonRow, ComparisonTable, Reconstruction } from "../types/api";

const METRIC_ORDER = ["clip_l14", "dinov2_base", "ssim", "lpips_alex"];

export function ComparePage() {
  return (
    <>
      <PageHeader title="Compare conditions & rate" description="Every generation of the active reference image, side by side. Nothing is hidden and nothing is auto-selected: rows are grouped by identical generation settings so only like is compared with like. Human ratings are stored separately from automatic metrics." />
      <WithReconstruction>{(r) => <CompareBody r={r} />}</WithReconstruction>
    </>
  );
}

function groupColor(g: string) {
  return CONDITION_COLOR[g] ?? (g.startsWith("C_") ? CONDITION_COLOR.C_refined : "#64748b");
}

function GroupPills({ row }: { row: ComparisonRow }) {
  if (!row.groups.length) return <span className="text-xs text-slate-500">unlabelled</span>;
  return (
    <span className="flex flex-wrap gap-1">
      {row.groups.map((g) => (
        <span key={g} className="inline-flex items-center gap-1.5 rounded-md border border-ink-600 bg-ink-850 px-1.5 py-0.5 text-[11px] text-slate-200">
          <span className="h-2 w-2 rounded-sm" style={{ background: groupColor(g) }} />{GROUP_LABEL[g] ?? g}
        </span>
      ))}
    </span>
  );
}

function CompareBody({ r }: { r: Reconstruction }) {
  const table = useApi(() => api.comparisonTable(r.id), [r.id, r.generations.length], { pollMs: 8000 });
  const [cfg, setCfg] = useState<string>("");
  const t = table.data;
  const groups = useMemo(() => Object.entries(t?.config_groups ?? {}).sort((a, b) => b[1].length - a[1].length), [t]);
  useEffect(() => { if (!cfg && groups.length) setCfg(groups[0][0]); }, [groups, cfg]);
  if (!t) return <Spinner />;
  if (!t.rows.length) return <Empty title="No generations yet">Run the controlled comparison in Stage 3 to populate this view.</Empty>;
  const rows = t.rows.filter((x) => cfg === "__all" || x.config_key === cfg);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end gap-3">
        <Field label="Generation settings (model | seed | steps | CFG | width | height)" className="min-w-80">
          <select className="input font-mono text-xs" value={cfg} onChange={(e) => setCfg(e.target.value)}>
            {groups.map(([k, ids]) => <option key={k} value={k}>{k} — {ids.length} run{ids.length > 1 ? "s" : ""}</option>)}
            <option value="__all">All settings (not directly comparable)</option>
          </select>
        </Field>
        {cfg === "__all" && <Pill tone="amber">mixed settings — differences may come from parameters, not prompts</Pill>}
      </div>

      <CompareTable t={t} rows={rows} referenceUrl={r.image.url} />

      <Notice tone="info" title="How to read this table">
        Arrows show each metric's direction (↑ higher = closer to the reference, ↓ lower = closer). CLIP and DINOv2 measure semantic similarity;
        SSIM compares aligned pixels and is expected to be low when the layout changes; LPIPS is a learned perceptual distance. “—” means the metric
        has not been computed (or failed) — it is never filled in. “Refined (best)” is chosen after the fact by the optimisation metric and is therefore
        biased; “Refined (final)” is what the loop actually ended with.
      </Notice>

      <RatingPanel r={r} rows={t.rows.filter((x) => x.status === "completed")} onSaved={table.reload} />
    </div>
  );
}

function CompareTable({ t, rows, referenceUrl }: { t: ComparisonTable; rows: ComparisonRow[]; referenceUrl: string }) {
  const metrics = METRIC_ORDER.filter((k) => t.metrics[k]);
  return (
    <Card pad={false} title="Results" subtitle={`${rows.length} generation${rows.length === 1 ? "" : "s"} · values are recorded measurements`}>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[1000px] text-sm">
          <thead className="text-left text-xs text-slate-400">
            <tr className="border-b border-ink-700">
              <th className="px-3 py-2.5 font-medium">Image</th>
              <th className="px-3 py-2.5 font-medium">Condition</th>
              <th className="px-3 py-2.5 font-medium">Prompt length</th>
              {metrics.map((k) => (
                <th key={k} className="px-3 py-2.5 text-right font-medium" title={t.metrics[k].higher_is_better ? "Higher is closer to the reference" : "Lower is closer to the reference"}>
                  <span className="inline-flex items-center gap-1">{t.metrics[k].label.replace(" similarity", "")}
                    {t.metrics[k].higher_is_better ? <ArrowUp className="h-3.5 w-3.5 text-slate-300" aria-label="higher is better" /> : <ArrowDown className="h-3.5 w-3.5 text-slate-300" aria-label="lower is better" />}</span>
                </th>
              ))}
              <th className="px-3 py-2.5 text-right font-medium">Gen. time</th>
              <th className="px-3 py-2.5 text-right font-medium">Refinement iter.</th>
              <th className="px-3 py-2.5 text-right font-medium" title="Mean human similarity rating, 1–5 (stored separately)">Human (1–5)</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-ink-800 bg-ink-850/40">
              <td className="px-3 py-2"><img src={referenceUrl} alt="Reference" className="h-14 w-14 rounded object-cover ring-1 ring-slate-500" /></td>
              <td className="px-3 py-2 text-slate-300" colSpan={3 + metrics.length + 2}>Reference image</td>
            </tr>
            {rows.map((x) => (
              <tr key={x.generation_id} className="border-b border-ink-800 align-top last:border-0">
                <td className="px-3 py-2"><GenThumb row={x} /></td>
                <td className="px-3 py-2"><GroupPills row={x} /><p className="mt-1 font-mono text-[11px] text-slate-500">v{x.prompt_version} · {x.template_id}</p></td>
                <td className="px-3 py-2 text-xs text-slate-300" title={x.prompt}>{x.prompt_words} words<br /><span className="text-slate-500">≈{x.prompt_clip_tokens_approx} CLIP tokens{x.prompt_clip_tokens_approx > 77 ? " (truncated at 77)" : ""}</span></td>
                {metrics.map((k) => {
                  const m = x.metrics[k];
                  return <td key={k} className="px-3 py-2 text-right font-mono text-slate-200" title={m ? m.status : "not requested"}>{m?.status === "completed" ? fmtNum(m.value, 3) : <span className="text-slate-600">{m ? `— ${m.status}` : "—"}</span>}</td>;
                })}
                <td className="px-3 py-2 text-right font-mono text-slate-300">{fmtDuration(x.generation_s)}{x.device && <span className="block text-[11px] text-slate-500">{x.device}</span>}</td>
                <td className="px-3 py-2 text-right font-mono text-slate-300">{x.refinement_iteration ?? "—"}{x.refinement_iterations_executed !== null && <span className="block text-[11px] text-slate-500">of {x.refinement_iterations_executed} run</span>}</td>
                <td className="px-3 py-2 text-right font-mono text-slate-300">{x.human.n ? <>{fmtNum(x.human.mean_similarity, 2)}<span className="block text-[11px] text-slate-500">n={x.human.n}</span></> : <span className="text-slate-600">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function GenThumb({ row }: { row: ComparisonRow }) {
  return row.output_image_url ? <img src={row.output_image_url} alt="" className="h-14 w-14 rounded object-cover" /> : <div className="flex h-14 w-14 items-center justify-center rounded bg-ink-800 text-[10px] text-slate-500">{row.status}</div>;
}

function shuffle<T>(xs: T[]): T[] {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

function RatingPanel({ r, rows, onSaved }: { r: Reconstruction; rows: ComparisonRow[]; onSaved: () => Promise<void> }) {
  const [blind, setBlind] = useState(true);
  const [order, setOrder] = useState<string[]>([]);
  const [idx, setIdx] = useState(0);
  const [rater, setRater] = useState(() => { try { return localStorage.getItem("pe.rater") ?? ""; } catch { return ""; } });
  const [sim, setSim] = useState<number | null>(null);
  const [subj, setSubj] = useState<number | null>(null);
  const [style, setStyle] = useState<number | null>(null);
  const [notes, setNotes] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(0);
  useEffect(() => { setOrder(shuffle(rows.map((x) => x.generation_id))); setIdx(0); }, [rows.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const current = rows.find((x) => x.generation_id === order[idx]);
  const url = current?.output_image_url;
  if (!rows.length) return null;

  const submit = async () => {
    if (!current || sim === null) return;
    setError(null);
    try {
      try { localStorage.setItem("pe.rater", rater); } catch { /* storage unavailable */ }
      await api.addRating(current.generation_id, { rater, similarity: sim, subject_match: subj, style_match: style, blind, notes });
      setSaved((n) => n + 1); setSim(null); setSubj(null); setStyle(null); setNotes("");
      setIdx((i) => (i + 1) % Math.max(order.length, 1));
      await onSaved();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };

  return (
    <Card title={<span className="flex items-center gap-2"><UserCheck className="h-4 w-4" />Human evaluation</span>}
      subtitle="Stored in its own table (human_ratings) and never mixed into automatic metrics. Blind mode hides prompts and condition labels and shows images in random order."
      actions={<div className="flex gap-2">
        <button className={clsx("btn-secondary", blind && "border-accent-400")} onClick={() => setBlind((b) => !b)}><EyeOff className="h-4 w-4" />{blind ? "Blind: on" : "Blind: off"}</button>
        <button className="btn-ghost" onClick={() => { setOrder(shuffle(rows.map((x) => x.generation_id))); setIdx(0); }}><Shuffle className="h-4 w-4" />Reshuffle</button>
      </div>}>
      <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
        <div className="grid grid-cols-2 gap-4">
          <ImageFrame src={r.image.url} label="Reference" />
          <ImageFrame src={url} label={blind ? `Candidate ${idx + 1} of ${order.length}` : current ? <GroupPills row={current} /> : "—"}
            sub={!blind && current ? <span className="mono line-clamp-2" title={current.prompt}>{current.prompt}</span> : undefined} />
        </div>
        <div className="space-y-3">
          <Field label="Rater"><input className="input" value={rater} onChange={(e) => setRater(e.target.value)} placeholder="initials" maxLength={64} /></Field>
          <Scale label="Overall similarity to the reference *" hint="1 unrelated · 3 same scene, clear differences · 5 near-identical" value={sim} onChange={setSim} />
          <Scale label="Subject match (optional)" value={subj} onChange={setSubj} />
          <Scale label="Style / colour match (optional)" value={style} onChange={setStyle} />
          <Field label="Notes"><textarea className="input min-h-16" value={notes} onChange={(e) => setNotes(e.target.value)} maxLength={2000} /></Field>
          <ErrorText error={error} />
          <div className="flex items-center gap-2">
            <button className="btn-primary flex-1" disabled={!rater.trim() || sim === null || !current} onClick={submit}><Star className="h-4 w-4" />Save rating & next</button>
            <button className="btn-ghost" onClick={() => setIdx((i) => (i + 1) % order.length)}>Skip</button>
          </div>
          {saved > 0 && <p className="text-xs text-emerald-300">{saved} rating{saved > 1 ? "s" : ""} saved this session.</p>}
        </div>
      </div>
    </Card>
  );
}

function Scale({ label, hint, value, onChange }: { label: string; hint?: string; value: number | null; onChange: (v: number | null) => void }) {
  return (
    <Field label={label} hint={hint}>
      <div className="flex gap-1.5" role="radiogroup" aria-label={label}>
        {[1, 2, 3, 4, 5].map((n) => (
          <button key={n} type="button" role="radio" aria-checked={value === n} onClick={() => onChange(value === n ? null : n)}
            className={clsx("h-9 flex-1 rounded-md border text-sm font-medium transition", value === n ? "border-accent-400 bg-accent-500/20 text-white" : "border-ink-600 bg-ink-850 text-slate-300 hover:border-ink-500")}>{n}</button>
        ))}
      </div>
    </Field>
  );
}
