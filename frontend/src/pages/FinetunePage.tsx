import clsx from "clsx";
import { ArrowDown, ArrowUp, Database, GraduationCap, Terminal } from "lucide-react";
import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, Empty, KV, Notice, Pill, Spinner, StatusBadge } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { PageHeader } from "../layouts/AppLayout";
import { AXIS_TEXT, GRID, TOOLTIP_STYLE } from "../lib/chart";
import { fmtDuration, fmtNum } from "../lib/format";
import { api } from "../services/api";
import type { FinetuneEval, FinetuneRun, FinetuneRunDetail } from "../types/api";

const TRAIN_COLOR = "#3987e5"; // categorical slot 1
const VAL_COLOR = "#d95926"; // categorical slot 2 (validated pair on the #0b1019 surface)
const METRIC_LABEL: Record<string, string> = { clip_l14: "CLIP ViT-L/14", dinov2_base: "DINOv2", ssim: "SSIM", lpips_alex: "LPIPS (AlexNet)" };
const RUNNING = new Set(["preparing", "training"]);

export function FinetunePage() {
  const runs = useApi(() => api.finetuneRuns(), [], { pollMs: 5000, shouldPoll: (rs) => rs.some((r) => RUNNING.has(r.status.state)) });
  const ds = useApi(() => api.finetuneDataset(), []);
  const [sel, setSel] = useState<string>("");
  useEffect(() => { if (!sel && runs.data?.length) setSel(runs.data[0].id); }, [runs.data, sel]);

  return (
    <>
      <PageHeader title="Fine-tuning (LoRA)" description="Low-rank adapters for the Stable Diffusion 1.5 UNet, trained on a filtered, CC0-licensed DiffusionDB subset and evaluated against the base model on held-out test prompts with the same metrics as the main study." />
      <div className="mb-6">
        <Notice tone="info" title="What this experiment can and cannot show">
          The adapter is trained on DiffusionDB images, which were themselves produced by Stable Diffusion 1.x from user prompts.
          A gain on its test split means the generator matches that distribution better — it does not by itself show better
          reconstruction of real photographs. Use the LoRA selector on Image Generation to test that on your own references.
        </Notice>
      </div>
      <div className="grid gap-6 xl:grid-cols-[380px_1fr]">
        <div className="space-y-6">
          <DatasetCard data={ds.data} />
          <Card title="Runs" subtitle="Training runs are never overwritten; each keeps its config, logs and adapter." pad={false}>
            {!runs.data ? <div className="p-5"><Spinner /></div> : runs.data.length === 0 ? <div className="p-5"><Empty title="No runs yet">Start one with the command below.</Empty></div> : (
              <ul className="divide-y divide-ink-800">
                {runs.data.map((r) => <RunRow key={r.id} r={r} active={r.id === sel} onClick={() => setSel(r.id)} />)}
              </ul>
            )}
          </Card>
          <Commands />
        </div>
        <div className="min-w-0 space-y-6">{sel ? <RunDetail id={sel} /> : null}</div>
      </div>
    </>
  );
}

function DatasetCard({ data }: { data: Awaited<ReturnType<typeof api.finetuneDataset>> | null }) {
  const c = data?.card;
  return (
    <Card title={<span className="flex items-center gap-2"><Database className="h-4 w-4" />Dataset</span>}
      subtitle={c ? "DiffusionDB 2M · CC0-1.0" : "Not built yet"}>
      {!data ? <Spinner /> : !c ? <p className="text-sm text-slate-400">Download parts with <code className="mono">scripts/download_diffusiondb.py</code>, then <code className="mono">python -m app.finetune.dataset build</code>.</p> : (
        <div className="space-y-3 text-sm">
          <div className="grid grid-cols-3 gap-2 text-center">
            {(["train", "val", "test"] as const).map((s) => (
              <div key={s} className="rounded-lg border border-ink-700 bg-ink-850 py-2">
                <p className="text-lg font-semibold tabular-nums text-slate-100">{c.counts[s]}</p>
                <p className="label">{s}</p>
              </div>
            ))}
          </div>
          <KV items={[
            ["Parts", c.parts.join(", ")],
            ["Candidates", c.candidates.toLocaleString()],
            ["Filters", `512×512 only · image & prompt NSFW < ${String(c.filters.max_image_nsfw)}`],
            ["Split", "grouped by normalised prompt (80/10/10)"],
            ["Prompt overlap", <span className="flex items-center gap-1.5">{Object.values(c.prompt_group_overlap).every((v) => v === 0) ? <Pill tone="emerald">none between splits</Pill> : <Pill tone="rose">overlap!</Pill>}</span>],
          ]} />
          <details className="text-xs text-slate-400">
            <summary className="cursor-pointer text-slate-300">Dropped during filtering</summary>
            <ul className="mt-1.5 space-y-0.5">{Object.entries(c.dropped).map(([k, v]) => <li key={k} className="flex justify-between"><span>{k}</span><span className="font-mono">{v}</span></li>)}</ul>
          </details>
        </div>
      )}
    </Card>
  );
}

function RunRow({ r, active, onClick }: { r: FinetuneRun; active: boolean; onClick: () => void }) {
  const s = r.status;
  const pct = s.total_steps ? Math.round((100 * s.step) / s.total_steps) : 0;
  return (
    <li>
      <button onClick={onClick} className={clsx("w-full px-5 py-3 text-left transition", active ? "bg-accent-500/10" : "hover:bg-ink-850")}>
        <div className="flex items-center justify-between gap-2">
          <span className="truncate font-mono text-sm text-slate-100">{r.id}</span>
          <StatusBadge status={s.state === "training" || s.state === "preparing" ? "running" : s.state} />
        </div>
        <p className="mt-1 text-xs text-slate-400">rank {r.config.rank} · lr {r.config.lr} · {r.config.steps} steps{r.evaluation ? " · evaluated" : ""}</p>
        {RUNNING.has(s.state) && (
          <div className="mt-2">
            <div className="h-1.5 overflow-hidden rounded-full bg-ink-700" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
              <div className="h-full bg-accent-400 transition-all" style={{ width: `${pct}%` }} />
            </div>
            <p className="mt-1 text-[11px] text-slate-500">{s.state === "preparing" ? s.message : `step ${s.step}/${s.total_steps} · ETA ${fmtDuration(s.eta_s ?? null)}`}</p>
          </div>
        )}
      </button>
    </li>
  );
}

function RunDetail({ id }: { id: string }) {
  const q = useApi(() => api.finetuneRun(id), [id], { pollMs: 5000, shouldPoll: (r) => RUNNING.has(r.status.state) });
  const ev = useApi(() => api.finetuneEval(id).catch(() => null), [id, q.data?.evaluation?.created_at]);
  const r = q.data;
  if (!r) return <Spinner />;
  const s = r.status;
  const valDelta = s.val_loss !== undefined && s.val_loss_initial !== undefined ? s.val_loss - s.val_loss_initial : null;
  return (
    <>
      <Card title={<span className="flex items-center gap-2"><GraduationCap className="h-4 w-4" /><span className="font-mono">{r.id}</span><StatusBadge status={RUNNING.has(s.state) ? "running" : s.state} /></span>}
        subtitle={r.config.method}>
        <div className="grid gap-4 md:grid-cols-4">
          <Stat label="Steps" value={`${s.step}/${s.total_steps}`} sub={`${r.config.grad_accum}× accumulation · ${fmtNum(s.epochs ?? null, 2)} epochs`} />
          <Stat label="Validation loss" value={fmtNum(s.val_loss ?? null, 4)} sub={valDelta === null ? "—" : `${valDelta <= 0 ? "▼" : "▲"} ${fmtNum(Math.abs(valDelta), 4)} vs start (${fmtNum(s.val_loss_initial ?? null, 4)})`} />
          <Stat label="Time" value={s.duration_s ? fmtDuration(s.duration_s) : s.eta_s ? `ETA ${fmtDuration(s.eta_s)}` : "—"} sub={s.sec_per_step ? `${s.sec_per_step} s/step on ${r.config.device}` : r.config.device} />
          <Stat label="Peak memory" value={`${fmtNum(s.memory_peak?.peak_process_gb ?? s.memory?.peak_process_gb ?? null, 2)} GB`} sub={`${r.config.frozen_dtype} UNet · fp32 LoRA`} />
        </div>
        <div className="mt-4 text-xs text-slate-400">
          rank {r.config.rank} · alpha {r.config.alpha} · lr {r.config.lr} · seed {r.config.seed} · {r.config.n_train} training images · {r.config.n_val_used} validation images
          {r.config.code_fingerprint?.git_commit && <> · code <span className="font-mono">{r.config.code_fingerprint.git_commit.slice(0, 7)}</span></>}
        </div>
        {s.state === "failed" && <div className="mt-3"><Notice tone="error">{s.message}</Notice></div>}
        {s.state === "stopped" && <div className="mt-3"><Notice tone="warn">Stopped before the last step; the adapter is not offered for generation.</Notice></div>}
      </Card>
      <LossChart r={r} />
      {ev.data ? <EvalResults e={ev.data} /> : (
        <Card title="Evaluation on held-out test prompts">
          <p className="text-sm text-slate-400">{s.state === "completed" ? <>Not evaluated yet. Run <code className="mono">python -m app.finetune.evaluate --run {r.id} --n 40</code> (≈ 15 min; generation then metrics, sequentially).</> : "Available after training completes."}</p>
        </Card>
      )}
    </>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-ink-700 bg-ink-850 p-3">
      <p className="label">{label}</p>
      <p className="mt-1 font-mono text-lg font-semibold tabular-nums text-slate-100">{value}</p>
      {sub && <p className="mt-0.5 text-[11px] text-slate-500">{sub}</p>}
    </div>
  );
}

function LossChart({ r }: { r: FinetuneRunDetail }) {
  const byStep = new Map<number, { step: number; train?: number; val?: number }>();
  for (const t of r.train_log) byStep.set(t.step, { step: t.step, train: t.loss_ema });
  for (const v of r.val_log) byStep.set(v.step, { ...(byStep.get(v.step) ?? { step: v.step }), val: v.val_loss });
  const rows = [...byStep.values()].sort((a, b) => a.step - b.step);
  if (rows.length < 2) return null;
  return (
    <Card title="Loss" subtitle="Training loss (exponential moving average, noisy by nature) and validation loss on fixed noise/timesteps — the comparable one.">
      <div className="h-64" role="img" aria-label="Training and validation loss by step">
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: -4 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} tick={{ fill: AXIS_TEXT, fontSize: 11 }} axisLine={{ stroke: GRID }} tickLine={false} />
            <YAxis tick={{ fill: AXIS_TEXT, fontSize: 11 }} axisLine={false} tickLine={false} domain={["auto", "auto"]} tickFormatter={(v) => fmtNum(v, 3)} width={56} />
            <Tooltip {...TOOLTIP_STYLE} formatter={(v: number, n) => [fmtNum(v, 5), n]} labelFormatter={(l) => `step ${l}`} />
            <Legend wrapperStyle={{ fontSize: 12, color: AXIS_TEXT }} />
            <Line name="training (EMA)" dataKey="train" stroke={TRAIN_COLOR} strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
            <Line name="validation" dataKey="val" stroke={VAL_COLOR} strokeWidth={2} dot={{ r: 4, fill: VAL_COLOR, stroke: "#0b1019", strokeWidth: 2 }} connectNulls isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

function EvalResults({ e }: { e: FinetuneEval }) {
  const [show, setShow] = useState(8);
  return (
    <>
      <Card title={<span className="flex items-center gap-2">Base vs LoRA on {e.n} held-out test prompts <Pill tone="emerald">Measured</Pill></span>}
        subtitle={`Same prompt, seed, ${String(e.settings.steps)} steps, CFG ${String(e.settings.cfg)}, 512×512 — only the adapter differs.`} pad={false}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[680px] text-sm">
            <thead className="text-left text-xs text-slate-400"><tr className="border-b border-ink-700">
              <th className="px-5 py-2.5 font-medium">Metric</th><th className="px-3 py-2.5 text-right font-medium">Base mean</th>
              <th className="px-3 py-2.5 text-right font-medium">LoRA mean</th><th className="px-3 py-2.5 text-right font-medium">Δ (LoRA − base)</th>
              <th className="px-3 py-2.5 text-right font-medium">LoRA closer</th><th className="px-3 py-2.5 text-right font-medium">LoRA further</th><th className="px-3 py-2.5 text-right font-medium">Ties</th>
            </tr></thead>
            <tbody className="text-slate-300">
              {Object.entries(e.paired).map(([k, p]) => {
                const improved = p.mean_diff_lora_minus_base !== null && (p.higher_is_better ? p.mean_diff_lora_minus_base > 0 : p.mean_diff_lora_minus_base < 0);
                return (
                  <tr key={k} className="border-b border-ink-800 last:border-0">
                    <td className="px-5 py-2"><span className="inline-flex items-center gap-1.5">{METRIC_LABEL[k] ?? k}{p.higher_is_better ? <ArrowUp className="h-3.5 w-3.5 text-slate-400" aria-label="higher is better" /> : <ArrowDown className="h-3.5 w-3.5 text-slate-400" aria-label="lower is better" />}</span></td>
                    <td className="px-3 py-2 text-right font-mono">{fmtNum(p.mean_base, 4)}</td>
                    <td className="px-3 py-2 text-right font-mono">{fmtNum(p.mean_lora, 4)}</td>
                    <td className="px-3 py-2 text-right font-mono">{fmtNum(p.mean_diff_lora_minus_base, 4)}<span className="ml-1.5 text-[11px] text-slate-500">{improved ? "closer" : "further"}</span></td>
                    <td className="px-3 py-2 text-right font-mono">{p.lora_better}</td>
                    <td className="px-3 py-2 text-right font-mono">{p.lora_worse}</td>
                    <td className="px-3 py-2 text-right font-mono">{p.ties}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="space-y-1 border-t border-ink-700 px-5 py-3 text-xs text-slate-400">
          {e.notes.map((n) => <p key={n}>• {n}</p>)}
          {(e.nsfw_flagged.base > 0 || e.nsfw_flagged.lora > 0) && <p>• Safety-checker flags (excluded from scoring): base {e.nsfw_flagged.base}, LoRA {e.nsfw_flagged.lora}.</p>}
        </div>
      </Card>
      <Card title="Samples" subtitle="Test items in manifest order (not selected by outcome). Reference · base · LoRA.">
        <div className="space-y-5">
          {e.records.slice(0, show).map((rec) => (
            <div key={rec.index}>
              <p className="mono mb-2 line-clamp-2 text-slate-300" title={rec.prompt}>#{rec.index} · {rec.prompt}</p>
              <div className="grid grid-cols-3 gap-3">
                {([["Reference (DiffusionDB)", rec.ref_file, null], ["Base SD 1.5", rec.base.file, rec.base], ["SD 1.5 + LoRA", rec.lora.file, rec.lora]] as const).map(([label, file, m]) => (
                  <figure key={label} className="overflow-hidden rounded-lg border border-ink-700 bg-ink-850">
                    <img src={api.finetuneFileUrl(e.run_id, file)} alt={label} loading="lazy" className="aspect-square w-full object-cover" />
                    <figcaption className="px-2 py-1.5 text-[11px] text-slate-400">
                      <span className="text-slate-200">{label}</span>
                      {m && <span className="block font-mono">CLIP-L {fmtNum(m.clip_l14 as number | null, 3)} · LPIPS {fmtNum(m.lpips_alex as number | null, 3)}</span>}
                    </figcaption>
                  </figure>
                ))}
              </div>
            </div>
          ))}
          {show < e.records.length && <button className="btn-secondary w-full" onClick={() => setShow((n) => n + 8)}>Show more ({e.records.length - show} left)</button>}
        </div>
      </Card>
    </>
  );
}

function Commands() {
  return (
    <Card title={<span className="flex items-center gap-2"><Terminal className="h-4 w-4" />Commands</span>}
      subtitle="Training and evaluation use most of the machine's memory for tens of minutes, so they run from a terminal (stop the local worker first).">
      <pre className="mono overflow-x-auto whitespace-pre rounded-lg border border-ink-700 bg-ink-950 p-3 text-[12px] leading-relaxed text-slate-300">{`cd backend
.venv/bin/python ../scripts/download_diffusiondb.py --parts 1 2 3
.venv/bin/python -m app.finetune.dataset build
.venv/bin/python -m app.finetune.train \\
    --name my-run --steps 500 --rank 8 --lr 1e-4
.venv/bin/python -m app.finetune.evaluate \\
    --run my-run --n 40`}</pre>
    </Card>
  );
}
