import clsx from "clsx";
import { Check, ChevronRight, Loader2 } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { PENDING, useApi } from "../hooks/useApi";
import { useActiveReconstruction } from "../hooks/useActiveReconstruction";
import { api } from "../services/api";
import type { Reconstruction } from "../types/api";
import { hasPending } from "./WithReconstruction";

type State = "done" | "running" | "next" | "todo";
interface Step { key: string; label: string; to: string; state: State; hint: string }

/** Derive the research workflow's progress for one reference image from its recorded rows only. */
export function workflowSteps(r: Reconstruction): Step[] {
  const ref = r.analyses.filter((a) => a.image_id === r.image.id);
  const anal = (mode: string) => ref.filter((a) => a.mode === mode);
  const st = (rows: { status: string }[]): "done" | "running" | "todo" =>
    rows.some((x) => x.status === "completed") ? "done" : rows.some((x) => PENDING.has(x.status)) ? "running" : "todo";
  const gens = r.generations;
  const conds = new Set(gens.filter((g) => g.status === "completed" && g.condition).map((g) => g.condition));
  const metrics = gens.flatMap((g) => g.metrics.filter((m) => m.role === "evaluation"));
  const refined = r.refinement_sessions.flatMap((s) => s.iterations.filter((i) => i.iteration > 0));
  const raw: (Omit<Step, "state"> & { s: "done" | "running" | "todo" })[] = [
    { key: "upload", label: "Upload", to: "/new", hint: "Reference image stored", s: "done" },
    { key: "caption", label: "Caption", to: "/analysis", hint: "VLM caption (condition A)", s: st(anal("caption")) },
    { key: "structured", label: "Analyse", to: "/analysis", hint: "Structured attributes (B, D)", s: st(anal("structured")) },
    { key: "prompts", label: "Prompts", to: "/prompts", hint: "Build, inspect and edit prompts", s: r.prompt_versions.length ? "done" : "todo" },
    { key: "generate", label: "Generate", to: "/generation", hint: "Images from the prompts", s: st(gens) },
    { key: "compare", label: "Compare", to: "/generation", hint: "≥ 2 conditions with identical settings",
      s: conds.size >= 2 ? "done" : gens.some((g) => PENDING.has(g.status) && g.condition) ? "running" : "todo" },
    { key: "evaluate", label: "Evaluate", to: "/evaluation", hint: "CLIP · DINOv2 · SSIM · LPIPS", s: st(metrics) },
    { key: "refine", label: "Refine", to: "/refinement", hint: "Iterative refinement session",
      s: refined.some((i) => i.status === "scored") ? "done" : refined.some((i) => PENDING.has(i.status)) || r.refinement_sessions.some((s) => s.status === "active") ? "running" : "todo" },
  ];
  let nextGiven = false;
  return raw.map(({ s, ...rest }) => {
    let state: State = s;
    if (s === "todo" && !nextGiven) { state = "next"; nextGiven = true; }
    return { ...rest, state };
  });
}

export function WorkflowStepper() {
  const { activeId } = useActiveReconstruction();
  const { search, pathname } = useLocation();
  const q = useApi(activeId ? () => api.reconstruction(activeId) : null, [activeId], { pollMs: 3000, shouldPoll: hasPending });
  if (!activeId || !q.data || pathname === "/present") return null;
  const steps = workflowSteps(q.data);
  const done = steps.filter((s) => s.state === "done").length;
  return (
    <nav aria-label="Research workflow progress" className="flex min-w-0 items-center gap-1 overflow-x-auto">
      <span className="mr-1 shrink-0 text-[11px] font-medium tabular-nums text-slate-500">{done}/{steps.length}</span>
      {steps.map((s, i) => (
        <div key={s.key} className="flex shrink-0 items-center">
          <Link to={{ pathname: s.to, search }} title={`${s.label}: ${s.hint}`}
            className={clsx("flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium transition",
              s.state === "done" && "border-emerald-500/30 bg-emerald-500/10 text-emerald-200 hover:bg-emerald-500/20",
              s.state === "running" && "border-accent-400/40 bg-accent-500/15 text-accent-200",
              s.state === "next" && "border-accent-400 bg-accent-500/25 text-white ring-2 ring-accent-500/30",
              s.state === "todo" && "border-ink-600 text-slate-500 hover:text-slate-300")}>
            {s.state === "done" ? <Check className="h-3 w-3" /> : s.state === "running" ? <Loader2 className="h-3 w-3 animate-spin" />
              : <span className="flex h-3 w-3 items-center justify-center text-[9px]">{i + 1}</span>}
            {s.label}
          </Link>
          {i < steps.length - 1 && <ChevronRight className="h-3 w-3 shrink-0 text-ink-500" />}
        </div>
      ))}
    </nav>
  );
}
