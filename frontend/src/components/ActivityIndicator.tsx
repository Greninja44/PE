import { Activity, Loader2 } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

const LABEL: Record<string, string> = { analyze: "Analysing", generate: "Generating", evaluate: "Evaluating" };

/** Live view of the inference queue: the running job with its progress, plus the number waiting. */
export function ActivityIndicator() {
  const { search } = useLocation();
  const running = useApi(() => api.jobs("running"), [], { pollMs: 2000 });
  const queued = useApi(() => api.jobs("queued"), [], { pollMs: 2000 });
  const job = running.data?.[0];
  const waiting = queued.data?.length ?? 0;
  if (!job && !waiting) {
    return <span className="hidden items-center gap-1.5 text-xs text-slate-500 md:flex"><Activity className="h-3.5 w-3.5" />Idle</span>;
  }
  const pct = job ? Math.round(job.progress * 100) : 0;
  return (
    <Link to={{ pathname: "/", search }} className="flex min-w-0 items-center gap-2 rounded-lg border border-accent-500/30 bg-accent-500/10 px-2.5 py-1 text-xs text-accent-200"
      title="Job queue (Overview)" aria-live="polite">
      <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
      {job ? (
        <>
          <span className="shrink-0 font-medium">{LABEL[job.type] ?? job.type}</span>
          <span className="hidden max-w-48 truncate text-accent-300/80 lg:inline">{job.progress_message}</span>
          <span className="h-1.5 w-16 shrink-0 overflow-hidden rounded-full bg-ink-700" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <span className="block h-full bg-accent-400 transition-all" style={{ width: `${pct}%` }} />
          </span>
          <span className="shrink-0 tabular-nums">{pct}%</span>
        </>
      ) : <span>Waiting for worker</span>}
      {waiting > 0 && <span className="shrink-0 text-accent-300/70">+{waiting} queued</span>}
    </Link>
  );
}
