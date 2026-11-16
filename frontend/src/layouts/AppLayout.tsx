import clsx from "clsx";
import {
  Beaker, BookOpen, Cpu, FlaskConical, Gauge, ImagePlus, Lightbulb, MonitorPlay, Repeat2, ScanSearch, Sparkles, Wand2,
} from "lucide-react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { ReconstructionPicker } from "../components/ReconstructionPicker";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

const NAV = [
  { group: "Lab", items: [
    { to: "/", label: "Overview", icon: Gauge },
    { to: "/new", label: "New Reconstruction", icon: ImagePlus },
  ] },
  { group: "Pipeline", items: [
    { to: "/analysis", label: "1 · Image Analysis", icon: ScanSearch },
    { to: "/prompts", label: "2 · Prompt Studio", icon: Wand2 },
    { to: "/generation", label: "3 · Image Generation", icon: Sparkles },
    { to: "/evaluation", label: "4 · Evaluation", icon: Beaker },
    { to: "/refinement", label: "5 · Refinement", icon: Repeat2 },
  ] },
  { group: "Research", items: [
    { to: "/experiments", label: "Experiments", icon: FlaskConical },
    { to: "/findings", label: "Research Findings", icon: Lightbulb },
    { to: "/methodology", label: "Methodology", icon: BookOpen },
  ] },
  { group: "Present", items: [{ to: "/present", label: "Presentation Mode", icon: MonitorPlay }] },
];

export function AppLayout() {
  const { search } = useLocation();
  const sys = useApi(() => api.system(), [], { pollMs: 10000 });
  const online = sys.data?.gpu_worker_online;
  const q = sys.data?.jobs;
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col border-r border-ink-700/70 bg-ink-900/60 lg:flex">
        <div className="flex items-center gap-2.5 px-5 py-5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-accent-500 to-cyan-500 shadow-lg shadow-accent-500/20">
            <ScanSearch className="h-5 w-5 text-white" />
          </div>
          <div className="leading-tight">
            <p className="text-sm font-semibold text-slate-100">Reconstruction Lab</p>
            <p className="text-[11px] text-slate-400">Image → Prompt research</p>
          </div>
        </div>
        <nav className="flex-1 space-y-5 overflow-y-auto px-3 pb-4">
          {NAV.map((g) => (
            <div key={g.group}>
              <p className="label mb-1.5 px-2.5 text-[10px]">{g.group}</p>
              <ul className="space-y-0.5">
                {g.items.map(({ to, label, icon: Icon }) => (
                  <li key={to}>
                    <NavLink to={{ pathname: to, search }} end={to === "/"}
                      className={({ isActive }) => clsx("flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition",
                        isActive ? "bg-accent-500/15 text-white ring-1 ring-inset ring-accent-500/30" : "text-slate-400 hover:bg-ink-800 hover:text-slate-200")}>
                      <Icon className="h-4 w-4" />{label}
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
        <div className="border-t border-ink-700/70 px-5 py-4 text-xs">
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-slate-400" />
            <span className="text-slate-300">GPU worker</span>
            <span className={clsx("ml-auto inline-flex items-center gap-1.5", online ? "text-emerald-300" : "text-slate-500")}>
              <span className={clsx("h-2 w-2 rounded-full", online ? "bg-emerald-400 shadow-[0_0_8px] shadow-emerald-400" : "bg-slate-600")} />
              {online ? "online" : "offline"}
            </span>
          </div>
          {q && <p className="mt-1.5 text-slate-500">{q.queued} queued · {q.running} running · {q.failed} failed</p>}
          {sys.data?.code && <p className="mono mt-1.5 truncate text-[11px] text-slate-600" title="code fingerprint">
            {sys.data.code.git_commit?.slice(0, 7) ?? "no-git"}{sys.data.code.git_dirty ? "+dirty" : ""} · src {sys.data.code.source_sha256}</p>}
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-ink-700/70 bg-ink-950/85 px-6 py-3 backdrop-blur">
          <ReconstructionPicker />
          <div className="ml-auto flex items-center gap-2 lg:hidden">
            <NavLink to={{ pathname: "/", search }} className="btn-ghost">Menu</NavLink>
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1500px] flex-1 px-6 py-6"><Outlet /></main>
      </div>
    </div>
  );
}

export function PageHeader({ title, description, actions, stage }: { title: string; description?: string; actions?: React.ReactNode; stage?: string }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        {stage && <p className="mb-1 text-xs font-semibold uppercase tracking-[0.14em] text-accent-300">{stage}</p>}
        <h1 className="text-2xl font-semibold tracking-tight text-white">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-slate-400">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}
