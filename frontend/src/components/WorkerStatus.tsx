import clsx from "clsx";
import { Cpu, HardDrive, MemoryStick } from "lucide-react";
import type { SystemInfo, WorkerInfo } from "../types/api";

const DEVICE_LABEL: Record<string, string> = { mps: "Apple MPS", cuda: "NVIDIA CUDA", cpu: "CPU" };

export function onlineWorker(sys?: SystemInfo | null): WorkerInfo | undefined {
  return sys?.workers.find((w) => w.online);
}

/** Compact status for the sidebar: device + live memory of the inference worker. */
export function WorkerBadge({ sys }: { sys?: SystemInfo | null }) {
  const w = onlineWorker(sys);
  const caps = w?.capabilities;
  const dev = caps?.runtime?.selected_device;
  const mem = caps?.memory;
  return (
    <div>
      <div className="flex items-center gap-2">
        <Cpu className="h-4 w-4 text-slate-400" />
        <span className="text-slate-300">Inference worker</span>
        <span className={clsx("ml-auto inline-flex items-center gap-1.5", w ? "text-emerald-300" : "text-slate-500")}>
          <span className={clsx("h-2 w-2 rounded-full", w ? "bg-emerald-400 shadow-[0_0_8px] shadow-emerald-400" : "bg-slate-600")} />
          {w ? (dev ? DEVICE_LABEL[dev] ?? dev : "online") : "offline"}
        </span>
      </div>
      {w && mem && (
        <p className="mt-1.5 text-slate-500" title="Worker process physical footprint (includes Metal/MPS buffers) · system memory available">
          worker {mem.process_gb?.toFixed(1)} GB · free {mem.system_available_gb?.toFixed(1)}/{mem.system_total_gb?.toFixed(0)} GB
        </p>
      )}
      {w && (caps?.resident_models?.length ?? 0) > 0 && (
        <p className="mt-0.5 truncate text-slate-500" title="Models currently resident in memory">
          loaded: {caps!.resident_models!.map((m) => m.key.split(":")[1]).join(", ")}
        </p>
      )}
    </div>
  );
}

/** Detailed model / memory status panel (Overview page). Shows only reported measurements. */
export function WorkerPanel({ sys }: { sys?: SystemInfo | null }) {
  const w = onlineWorker(sys);
  if (!sys) return null;
  const caps = w?.capabilities;
  const rt = caps?.runtime;
  const mem = caps?.memory;
  const usedPct = mem?.system_total_gb && mem.system_available_gb !== undefined
    ? Math.round(100 * (1 - mem.system_available_gb / mem.system_total_gb)) : null;
  return (
    <div className="space-y-4 text-sm">
      {!w ? (
        <p className="text-slate-400">No inference worker is online. Start everything with <code className="mono">scripts/dev.sh</code>, or the
          worker alone with <code className="mono">backend/.venv/bin/python -m app.workers.agent local</code>. Jobs stay queued until a worker connects.</p>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            <Stat label="Device" value={rt?.selected_device ? DEVICE_LABEL[rt.selected_device] ?? rt.selected_device : "—"} sub={rt?.gpu ?? undefined} />
            <Stat label="Worker memory" value={mem?.process_gb !== undefined ? `${mem.process_gb.toFixed(2)} GB` : "—"}
              sub={mem?.process_metric === "phys_footprint" ? "physical footprint incl. GPU buffers" : mem?.process_metric} />
            <Stat label="System memory free" value={mem?.system_available_gb !== undefined ? `${mem.system_available_gb.toFixed(1)} GB` : "—"}
              sub={mem?.system_total_gb ? `of ${mem.system_total_gb.toFixed(0)} GB${mem.swap_used_gb ? ` · swap ${mem.swap_used_gb.toFixed(1)} GB` : ""}` : undefined} />
          </div>
          {usedPct !== null && (
            <div>
              <div className="h-2 overflow-hidden rounded-full bg-ink-800" role="meter" aria-valuenow={usedPct} aria-valuemin={0} aria-valuemax={100} aria-label="System memory in use">
                <div className={clsx("h-full rounded-full", usedPct > 85 ? "bg-rose-400" : usedPct > 70 ? "bg-amber-400" : "bg-accent-400")} style={{ width: `${usedPct}%` }} />
              </div>
              <p className="mt-1 text-xs text-slate-500">{usedPct}% of system memory in use. Models load one family at a time (VLM → diffusion → metrics) and unload when idle.</p>
            </div>
          )}
          <div>
            <p className="label mb-1.5">Resident models</p>
            {caps?.resident_models?.length ? (
              <ul className="space-y-1">{caps.resident_models.map((m) => (
                <li key={m.key} className="flex items-center gap-2 text-xs text-slate-300"><MemoryStick className="h-3.5 w-3.5 text-slate-500" />
                  <span className="font-mono">{m.key}</span><span className="text-slate-500">loaded in {m.load_s}s · idle {Math.round(m.idle_s)}s</span></li>
              ))}</ul>
            ) : <p className="text-xs text-slate-500">None — memory released.</p>}
          </div>
        </>
      )}
      <div>
        <p className="label mb-1.5">Model files on this machine</p>
        <ul className="grid gap-1 text-xs sm:grid-cols-2">
          {Object.entries(sys.local_vlm_cached ?? {}).filter(([k]) => k.includes("Qwen3-VL-2B")).map(([k, v]) => <FileRow key={k} name={k} ok={v} />)}
          {sys.models.map((m) => <FileRow key={m.key} name={`${m.label}${m.recommended_16gb === false ? " (not recommended on 16 GB)" : ""}`} ok={m.downloaded} />)}
        </ul>
        <p className="mt-1.5 text-xs text-slate-500">Metric models (CLIP, DINOv2, LPIPS) download on first use. Fetch everything in advance with <code className="mono">scripts/download_models.py</code>.</p>
      </div>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-ink-700 bg-ink-850 p-3">
      <p className="label">{label}</p>
      <p className="mt-1 text-lg font-semibold text-slate-100">{value}</p>
      {sub && <p className="truncate text-xs text-slate-500" title={sub}>{sub}</p>}
    </div>
  );
}

function FileRow({ name, ok }: { name: string; ok?: boolean | null }) {
  return (
    <li className="flex items-center gap-2 text-slate-300">
      <HardDrive className={clsx("h-3.5 w-3.5", ok ? "text-emerald-400" : "text-slate-600")} />
      <span className="truncate">{name}</span>
      <span className={clsx("ml-auto shrink-0", ok ? "text-emerald-300" : "text-slate-500")}>{ok ? "downloaded" : "not downloaded"}</span>
    </li>
  );
}
