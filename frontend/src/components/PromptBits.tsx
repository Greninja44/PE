import clsx from "clsx";
import { useApi } from "../hooks/useApi";
import { FIELD_LABEL } from "../lib/format";
import { api } from "../services/api";
import type { DiffOp, PromptVersion, Validation } from "../types/api";
import { Pill, Spinner } from "./ui";

const SOURCE_TONE: Record<string, "slate" | "accent" | "cyan" | "amber" | "emerald" | "rose"> = {
  caption: "slate", structured: "cyan", validated: "emerald", edited: "accent", manual: "amber", refinement: "amber",
};

export function SourcePill({ source }: { source: string }) {
  return <Pill tone={SOURCE_TONE[source] ?? "slate"}>{source}</Pill>;
}

export function TokenMeter({ v }: { v: Validation }) {
  const usable = v.token_limit - 2;
  const pct = Math.min(100, (v.approx_clip_tokens / usable) * 100);
  const over = v.approx_clip_tokens > usable;
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-slate-400">CLIP window (approx.)</span>
        <span className={clsx("font-mono", over ? "text-amber-300" : "text-slate-300")}>≈{v.approx_clip_tokens} / {usable} tokens</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-ink-700">
        <div className={clsx("h-full rounded-full", over ? "bg-amber-400" : "bg-gradient-to-r from-accent-500 to-cyan-400")} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

const FIELD_COLOR: Record<string, string> = {
  subject: "border-accent-400/50 bg-accent-500/10", art_style: "border-fuchsia-400/40 bg-fuchsia-500/10",
  secondary_objects: "border-cyan-400/40 bg-cyan-500/10", spatial_relationships: "border-sky-400/40 bg-sky-500/10",
  lighting: "border-amber-400/40 bg-amber-500/10", colors: "border-rose-400/40 bg-rose-500/10",
  background: "border-emerald-400/40 bg-emerald-500/10",
};

export function SegmentedPrompt({ segments }: { segments: { field: string; text: string }[] }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {segments.map((s, i) => (
        <span key={i} title={FIELD_LABEL[s.field] ?? s.field}
          className={clsx("rounded-md border px-1.5 py-0.5 text-[13px] text-slate-100", FIELD_COLOR[s.field] ?? "border-ink-600 bg-ink-800")}>
          {s.text}
        </span>
      ))}
    </div>
  );
}

export function PromptText({ label, text }: { label: string; text: string }) {
  return (
    <div>
      <p className="label mb-1">{label}</p>
      <p className="mono whitespace-pre-wrap rounded-lg border border-ink-700 bg-ink-950/70 p-3 leading-relaxed text-slate-200">{text || <span className="text-slate-600">(empty)</span>}</p>
    </div>
  );
}

function Ops({ ops }: { ops: DiffOp[] }) {
  return (
    <p className="mono whitespace-pre-wrap rounded-lg border border-ink-700 bg-ink-950/70 p-3 leading-relaxed">
      {ops.map((o, i) => (
        <span key={i} className={clsx(o.op === "insert" && "rounded bg-emerald-500/20 text-emerald-200",
          o.op === "delete" && "rounded bg-rose-500/20 text-rose-300 line-through", o.op === "equal" && "text-slate-400")}>
          {o.text}{" "}
        </span>
      ))}
    </p>
  );
}

export function PromptDiffView({ a, b }: { a: string; b: string }) {
  const d = useApi(() => api.promptDiff(a, b), [a, b]);
  if (!d.data) return <Spinner />;
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-400">v{d.data.a.version} → v{d.data.b.version}</p>
      <div><p className="label mb-1">Positive</p><Ops ops={d.data.positive} /></div>
      <div><p className="label mb-1">Negative</p><Ops ops={d.data.negative} /></div>
      {d.data.attributes.length > 0 && (
        <div>
          <p className="label mb-1">Attribute changes</p>
          <ul className="space-y-1 text-sm">
            {d.data.attributes.map((c) => (
              <li key={c.field} className="rounded-md border border-ink-700 bg-ink-850 px-3 py-1.5">
                <span className="text-slate-300">{FIELD_LABEL[c.field] ?? c.field}: </span>
                <span className="text-rose-300 line-through">{JSON.stringify(c.before)}</span> → <span className="text-emerald-300">{JSON.stringify(c.after)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function versionLabel(p: PromptVersion) {
  return `v${p.version} · ${p.source}${p.validation && !p.validation.ok ? " ⚠" : ""}`;
}
