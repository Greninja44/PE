import clsx from "clsx";
import { AlertTriangle, CheckCircle2, CircleDashed, Clock, Info, Loader2, XCircle } from "lucide-react";
import type { ReactNode } from "react";
import { fmtNum } from "../lib/format";
import type { Issue, MetricResult } from "../types/api";

export function Card({ title, subtitle, actions, children, className, pad = true }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; pad?: boolean;
}) {
  return (
    <section className={clsx("card", className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-ink-700/70 px-5 py-3.5">
          <div>
            {title && <h2 className="text-[15px] font-semibold text-slate-100">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-[13px] text-slate-400">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={clsx(pad && "card-pad")}>{children}</div>
    </section>
  );
}

const STATUS_STYLE: Record<string, string> = {
  completed: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  succeeded: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  scored: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  queued: "border-amber-500/30 bg-amber-500/10 text-amber-200",
  pending: "border-amber-500/30 bg-amber-500/10 text-amber-200",
  proposed: "border-sky-500/30 bg-sky-500/10 text-sky-200",
  approved: "border-sky-500/30 bg-sky-500/10 text-sky-200",
  running: "border-accent-400/40 bg-accent-500/15 text-accent-300",
  generating: "border-accent-400/40 bg-accent-500/15 text-accent-300",
  scoring: "border-accent-400/40 bg-accent-500/15 text-accent-300",
  active: "border-accent-400/40 bg-accent-500/15 text-accent-300",
  failed: "border-rose-500/30 bg-rose-500/10 text-rose-300",
  cancelled: "border-slate-500/30 bg-slate-500/10 text-slate-300",
  rejected: "border-slate-500/30 bg-slate-500/10 text-slate-300",
  unavailable: "border-slate-500/30 bg-slate-500/10 text-slate-300",
  not_executed: "border-slate-500/30 bg-slate-500/10 text-slate-400",
  stopped_early: "border-teal-500/30 bg-teal-500/10 text-teal-200",
  max_iterations: "border-teal-500/30 bg-teal-500/10 text-teal-200",
  stopped: "border-slate-500/30 bg-slate-500/10 text-slate-300",
};

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const Icon = ["completed", "succeeded", "scored"].includes(status) ? CheckCircle2
    : ["failed"].includes(status) ? XCircle
    : ["running", "generating", "scoring"].includes(status) ? Loader2
    : ["queued", "pending"].includes(status) ? Clock : CircleDashed;
  return (
    <span className={clsx("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium",
      STATUS_STYLE[status] ?? STATUS_STYLE.cancelled, className)}>
      <Icon className={clsx("h-3 w-3", ["running", "generating", "scoring"].includes(status) && "animate-spin")} />
      {status.replace(/_/g, " ")}
    </span>
  );
}

export function Pill({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "accent" | "cyan" | "amber" | "rose" | "emerald" }) {
  const tones = {
    slate: "border-ink-600 bg-ink-800 text-slate-300",
    accent: "border-accent-500/40 bg-accent-500/10 text-accent-300",
    cyan: "border-cyan-500/30 bg-cyan-500/10 text-cyan-200",
    amber: "border-amber-500/30 bg-amber-500/10 text-amber-200",
    rose: "border-rose-500/30 bg-rose-500/10 text-rose-200",
    emerald: "border-emerald-500/30 bg-emerald-500/10 text-emerald-200",
  };
  return <span className={clsx("inline-flex items-center rounded-md border px-1.5 py-0.5 text-[11px] font-medium", tones[tone])}>{children}</span>;
}

export function Empty({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-ink-600 px-6 py-10 text-center">
      {icon && <div className="mb-3 text-slate-500">{icon}</div>}
      <p className="font-medium text-slate-200">{title}</p>
      {children && <div className="mt-1.5 max-w-lg text-sm text-slate-400">{children}</div>}
    </div>
  );
}

export function Notice({ tone = "info", children, title }: { tone?: "info" | "warn" | "error" | "ok"; title?: string; children: ReactNode }) {
  const map = {
    info: ["border-sky-500/30 bg-sky-500/5 text-sky-100", Info],
    warn: ["border-amber-500/30 bg-amber-500/5 text-amber-100", AlertTriangle],
    error: ["border-rose-500/30 bg-rose-500/5 text-rose-100", XCircle],
    ok: ["border-emerald-500/30 bg-emerald-500/5 text-emerald-100", CheckCircle2],
  } as const;
  const [cls, Icon] = map[tone];
  return (
    <div className={clsx("flex gap-3 rounded-lg border px-4 py-3 text-sm", cls)}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" />
      <div>{title && <p className="mb-0.5 font-semibold">{title}</p>}<div className="opacity-90">{children}</div></div>
    </div>
  );
}

export function ErrorText({ error }: { error: string | null }) {
  if (!error) return null;
  return <Notice tone="error">{error}</Notice>;
}

export function Field({ label, hint, children, className }: { label: string; hint?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <label className={clsx("block", className)}>
      <span className="label mb-1.5 block">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-slate-500">{hint}</span>}
    </label>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={clsx("h-4 w-4 animate-spin text-slate-400", className)} />;
}

export function KV({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1.5 text-sm">
      {items.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-slate-400">{k}</dt>
          <dd className="min-w-0 break-words text-slate-200">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** A metric value that NEVER renders a number unless the metric actually completed. */
export function MetricTile({ m, large }: { m: MetricResult; large?: boolean }) {
  const done = m.status === "completed" && m.value !== null;
  return (
    <div className="rounded-lg border border-ink-700 bg-ink-850 p-3.5">
      <div className="flex items-start justify-between gap-2">
        <p className="text-[13px] font-medium text-slate-300">{m.label}</p>
        <Pill tone={m.role === "optimization" ? "amber" : "cyan"}>{m.role === "optimization" ? "optimisation" : "evaluation"}</Pill>
      </div>
      <p className={clsx("mt-2 font-mono tabular-nums", large ? "text-3xl" : "text-2xl", done ? "text-slate-50" : "text-slate-500")}>
        {done ? fmtNum(m.value) : "—"}
      </p>
      <div className="mt-1.5 flex items-center justify-between gap-2">
        <StatusBadge status={m.status} />
        <span className="text-[11px] text-slate-500">{m.higher_is_better ? "↑ higher = closer" : "↓ lower = closer"}</span>
      </div>
      {!done && m.detail && <p className="mt-2 line-clamp-2 text-[11px] text-slate-500" title={m.detail}>{m.detail}</p>}
      {done && <p className="mt-2 truncate text-[11px] text-slate-500" title={m.backend}>{m.backend}</p>}
    </div>
  );
}

export function IssueList({ issues }: { issues: Issue[] }) {
  if (!issues.length) return <Notice tone="ok">No validation issues.</Notice>;
  const icon = { error: <XCircle className="h-4 w-4 text-rose-400" />, warning: <AlertTriangle className="h-4 w-4 text-amber-400" />, info: <Info className="h-4 w-4 text-sky-400" /> };
  return (
    <ul className="space-y-1.5">
      {issues.map((i, n) => (
        <li key={n} className="flex gap-2.5 rounded-md border border-ink-700 bg-ink-850 px-3 py-2 text-sm">
          <span className="mt-0.5 shrink-0">{icon[i.severity]}</span>
          <div className="min-w-0">
            <p className="text-slate-200">{i.message}</p>
            <p className="mono mt-0.5 text-[11px] text-slate-500">{i.code}{i.fields.length ? ` · ${i.fields.join(", ")}` : ""}{i.auto_fixable ? " · auto-fixable" : ""}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}

export function ImageFrame({ src, label, sub, className }: { src?: string | null; label?: ReactNode; sub?: ReactNode; className?: string }) {
  return (
    <figure className={clsx("overflow-hidden rounded-xl border border-ink-700 bg-ink-850", className)}>
      <div className="flex aspect-square items-center justify-center bg-[repeating-conic-gradient(#0f1622_0_25%,#121a28_0_50%)] bg-[length:20px_20px]">
        {src ? <img src={src} alt={typeof label === "string" ? label : ""} className="h-full w-full object-contain" />
          : <span className="px-4 text-center text-sm text-slate-500">Not generated</span>}
      </div>
      {(label || sub) && (
        <figcaption className="border-t border-ink-700 px-3 py-2">
          {label && <p className="text-sm font-medium text-slate-200">{label}</p>}
          {sub && <div className="mt-0.5 text-xs text-slate-400">{sub}</div>}
        </figcaption>
      )}
    </figure>
  );
}

export function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between"><h3 className="label">{title}</h3>{aside}</div>
      {children}
    </div>
  );
}
