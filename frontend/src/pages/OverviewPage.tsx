import { ArrowRight, Cpu, RotateCcw, ScanSearch, Sparkles, Beaker, Repeat2, Wand2, XCircle } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { PageHeader } from "../layouts/AppLayout";
import { useApi } from "../hooks/useApi";
import { useActiveReconstruction } from "../hooks/useActiveReconstruction";
import { fmtTime, shortId } from "../lib/format";
import { api } from "../services/api";
import { Card, Empty, Notice, Pill, StatusBadge } from "../components/ui";

const STAGES = [
  { n: 1, title: "Image Analysis", icon: ScanSearch, to: "/analysis", text: "Qwen3-VL extracts structured attributes as JSON." },
  { n: 2, title: "Prompt Studio", icon: Wand2, to: "/prompts", text: "Fixed templates → positive & negative prompts, validated." },
  { n: 3, title: "Generation", icon: Sparkles, to: "/generation", text: "SD 1.5 · SD-Turbo · SDXL with explicit seeds & params." },
  { n: 4, title: "Evaluation", icon: Beaker, to: "/evaluation", text: "CLIP · DINOv2 · SSIM · LPIPS, reported separately." },
  { n: 5, title: "Refinement", icon: Repeat2, to: "/refinement", text: "Proposals → approve/reject → early stopping." },
];

export function OverviewPage() {
  const { search } = useLocation();
  const { setActiveId } = useActiveReconstruction();
  const sys = useApi(() => api.system(), [], { pollMs: 5000 });
  const recons = useApi(() => api.reconstructions(), []);
  const jobs = useApi(() => api.jobs(), [], { pollMs: 4000 });
  const s = sys.data;

  return (
    <>
      <PageHeader title="Image-to-Prompt Reconstruction" description="Reverse-engineer a text prompt that regenerates a reference image, and measure how close each pipeline stage gets. Every number on this dashboard comes from a recorded run — nothing is simulated." />

      <div className="mb-6 grid gap-3 md:grid-cols-5">
        {STAGES.map(({ n, title, icon: Icon, to, text }) => (
          <Link key={n} to={{ pathname: to, search }} className="card group relative overflow-hidden p-4 transition hover:border-accent-500/50">
            <div className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-accent-500/0 via-accent-400/70 to-cyan-400/0 opacity-0 transition group-hover:opacity-100" />
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-md bg-accent-500/15 text-xs font-semibold text-accent-300">{n}</span>
              <Icon className="h-4 w-4 text-slate-400" />
            </div>
            <p className="mt-3 font-medium text-slate-100">{title}</p>
            <p className="mt-1 text-[13px] leading-snug text-slate-400">{text}</p>
          </Link>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card title="Compute & providers" subtitle="Availability is checked live; unavailable stages are refused, never simulated." className="xl:col-span-1">
          {s ? (
            <div className="space-y-4 text-sm">
              <div className="flex items-center gap-3 rounded-lg border border-ink-700 bg-ink-850 p-3">
                <Cpu className={s.gpu_worker_online ? "h-5 w-5 text-emerald-400" : "h-5 w-5 text-slate-500"} />
                <div className="min-w-0 flex-1">
                  <p className="font-medium text-slate-100">GPU worker {s.gpu_worker_online ? "online" : "offline"}</p>
                  <p className="truncate text-xs text-slate-400">
                    {s.workers.length ? s.workers.map((w) => `${w.name}${w.online ? "" : " (idle)"}`).join(", ") : "No worker registered yet"}
                  </p>
                </div>
              </div>
              {!s.gpu_worker_online && <Notice tone="warn">Worker jobs (Qwen3-VL, diffusion, CLIP/DINOv2/LPIPS) will wait in the queue until a GPU worker connects. See <span className="mono">docs/RTX5060_SETUP.md</span>.</Notice>}
              {[["Vision-language", s.vlm_providers], ["Generation", s.generation_providers]].map(([label, list]) => (
                <div key={label as string}>
                  <p className="label mb-1.5">{label as string}</p>
                  <ul className="space-y-1">
                    {(list as typeof s.vlm_providers).map((p) => (
                      <li key={p.name} className="flex items-center justify-between gap-2">
                        <span className="mono text-slate-300">{p.name}</span>
                        <span className="flex items-center gap-1.5">
                          <Pill tone={p.execution === "worker" ? "accent" : "slate"}>{p.execution}</Pill>
                          <Pill tone={p.available ? "emerald" : "rose"}>{p.available ? "ready" : "not configured"}</Pill>
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
              <div>
                <p className="label mb-1.5">Metrics</p>
                <ul className="space-y-1">
                  {s.metrics.map((m) => (
                    <li key={m.key} className="flex items-center justify-between gap-2">
                      <span className="text-slate-300">{m.label}</span>
                      <Pill tone={m.execution === "inline" ? "emerald" : "accent"}>{m.execution === "inline" ? "runs here" : "GPU worker"}</Pill>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : <p className="text-sm text-slate-400">Connecting to backend…</p>}
        </Card>

        <Card title="Reconstructions" subtitle="Reference images under study" className="xl:col-span-2"
          actions={<Link to={{ pathname: "/new", search }} className="btn-primary">New reconstruction</Link>}>
          {recons.data?.length ? (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {recons.data.slice(0, 9).map((r) => (
                <button key={r.id} onClick={() => setActiveId(r.id)} className="group flex gap-3 rounded-lg border border-ink-700 bg-ink-850 p-2.5 text-left transition hover:border-accent-500/50">
                  <img src={r.image.url} alt="" className="h-16 w-16 shrink-0 rounded-md object-cover" />
                  <div className="min-w-0">
                    <p className="truncate font-medium text-slate-100">{r.name}</p>
                    <p className="text-xs text-slate-400">{r.n_prompts} prompts · {r.n_completed_generations}/{r.n_generations} images</p>
                    <p className="mt-1 flex items-center gap-1 text-xs text-accent-300 opacity-0 transition group-hover:opacity-100">Make active <ArrowRight className="h-3 w-3" /></p>
                  </div>
                </button>
              ))}
            </div>
          ) : <Empty title="No reconstructions yet">Upload a reference image to begin.</Empty>}
        </Card>

        <Card title="Job queue" subtitle="Work dispatched to the GPU worker" className="xl:col-span-3" pad={false}>
          {jobs.data?.length ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-slate-400">
                  <tr className="border-b border-ink-700">{["Job", "Type", "Status", "Progress", "Attempts", "Created", "Error", ""].map((h) => <th key={h} className="px-4 py-2.5 font-medium">{h}</th>)}</tr>
                </thead>
                <tbody>
                  {jobs.data.slice(0, 25).map((j) => (
                    <tr key={j.id} className="border-b border-ink-800 last:border-0">
                      <td className="mono px-4 py-2 text-slate-300">{shortId(j.id)}</td>
                      <td className="px-4 py-2">{j.type}</td>
                      <td className="px-4 py-2"><StatusBadge status={j.status} /></td>
                      <td className="px-4 py-2">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-24 overflow-hidden rounded-full bg-ink-700"><div className="h-full bg-accent-400" style={{ width: `${Math.round(j.progress * 100)}%` }} /></div>
                          <span className="truncate text-xs text-slate-400">{j.progress_message}</span>
                        </div>
                      </td>
                      <td className="px-4 py-2 text-slate-400">{j.attempts}/{j.max_attempts}</td>
                      <td className="px-4 py-2 text-slate-400">{fmtTime(j.created_at)}</td>
                      <td className="max-w-xs truncate px-4 py-2 text-xs text-rose-300" title={j.error ?? ""}>{j.error}</td>
                      <td className="px-4 py-2 text-right">
                        {["queued", "running"].includes(j.status) && <button className="btn-ghost px-2 py-1" title="Cancel" onClick={() => api.cancelJob(j.id).then(jobs.reload)}><XCircle className="h-4 w-4" /></button>}
                        {["failed", "cancelled"].includes(j.status) && <button className="btn-ghost px-2 py-1" title="Retry" onClick={() => api.retryJob(j.id).then(jobs.reload)}><RotateCcw className="h-4 w-4" /></button>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div className="p-5"><Empty title="Queue empty">Jobs appear here when you request worker-executed analysis, generation or metrics.</Empty></div>}
        </Card>
      </div>
    </>
  );
}
