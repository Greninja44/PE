import clsx from "clsx";
import {
  Beaker, BookOpen, Columns3, FlaskConical, GraduationCap, Menu, X, Gauge, ImagePlus, Lightbulb, MonitorPlay, Repeat2, ScanSearch, Sparkles, Wand2,
} from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { ReconstructionPicker } from "../components/ReconstructionPicker";
import { ActivityIndicator } from "../components/ActivityIndicator";
import { WorkerBadge } from "../components/WorkerStatus";
import { WorkflowStepper } from "../components/WorkflowStepper";
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
    { to: "/compare", label: "Compare & Rate", icon: Columns3 },
    { to: "/experiments", label: "Experiments", icon: FlaskConical },
    { to: "/finetune", label: "Fine-tuning (LoRA)", icon: GraduationCap },
    { to: "/findings", label: "Research Findings", icon: Lightbulb },
    { to: "/methodology", label: "Methodology", icon: BookOpen },
  ] },
  { group: "Present", items: [{ to: "/present", label: "Presentation Mode", icon: MonitorPlay }] },
];

export function AppLayout() {
  const { search, pathname } = useLocation();
  const sys = useApi(() => api.system(), [], { pollMs: 10000 });
  const [menu, setMenu] = useState(false);
  useEffect(() => setMenu(false), [pathname]);
  useEffect(() => {
    if (!menu) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setMenu(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menu]);
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
        <NavList search={search} />
        <div className="border-t border-ink-700/70 px-5 py-4 text-xs">
          <WorkerBadge sys={sys.data} />
          {q && <p className="mt-1.5 text-slate-500">{q.queued} queued · {q.running} running · {q.failed} failed</p>}
          {sys.data?.code && <p className="mono mt-1.5 truncate text-[11px] text-slate-600" title="code fingerprint">
            {sys.data.code.git_commit?.slice(0, 7) ?? "no-git"}{sys.data.code.git_dirty ? "+dirty" : ""} · src {sys.data.code.source_sha256}</p>}
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-ink-700/70 bg-ink-950/85 px-4 py-3 backdrop-blur sm:px-6">
          <button className="btn-ghost px-2 lg:hidden" onClick={() => setMenu(true)} aria-label="Open navigation" aria-expanded={menu}><Menu className="h-5 w-5" /></button>
          <ReconstructionPicker />
          <div className="ml-auto flex items-center gap-2">
            <ActivityIndicator />
          </div>
        </header>
        <div className="border-b border-ink-700/70 bg-ink-900/40 px-4 py-2 empty:hidden sm:px-6"><WorkflowStepper /></div>
        <main className="mx-auto w-full max-w-[1500px] flex-1 px-4 py-6 sm:px-6"><Outlet /></main>
      </div>
      {menu && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <button className="absolute inset-0 bg-black/60" aria-label="Close navigation" onClick={() => setMenu(false)} />
          <aside className="absolute inset-y-0 left-0 flex w-72 max-w-[85vw] flex-col border-r border-ink-700 bg-ink-900 shadow-2xl">
            <div className="flex items-center justify-between px-5 py-4">
              <p className="text-sm font-semibold text-slate-100">Reconstruction Lab</p>
              <button className="btn-ghost px-2" onClick={() => setMenu(false)} aria-label="Close navigation"><X className="h-5 w-5" /></button>
            </div>
            <NavList search={search} />
            <div className="border-t border-ink-700/70 px-5 py-4 text-xs"><WorkerBadge sys={sys.data} /></div>
          </aside>
        </div>
      )}
    </div>
  );
}

function NavList({ search }: { search: string }) {
  return (
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
