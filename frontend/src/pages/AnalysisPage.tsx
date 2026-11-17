import { ChevronDown, Loader2, PenLine, Play, Sparkles, Wand2 } from "lucide-react";
import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { AttributeEditor, AttributeView } from "../components/AttributeEditor";
import { Card, Empty, ErrorText, Field, ImageFrame, KV, Notice, Pill, StatusBadge } from "../components/ui";
import { WithReconstruction } from "../components/WithReconstruction";
import { useApi } from "../hooks/useApi";
import { PageHeader } from "../layouts/AppLayout";
import { emptyAttributes, fmtDuration, fmtTime } from "../lib/format";
import { api } from "../services/api";
import type { Analysis, Attributes, Reconstruction } from "../types/api";

export function AnalysisPage() {
  return (
    <>
      <PageHeader stage="Stage 1" title="Image analysis" description="A vision-language model describes the reference image as structured JSON. Qwen3-VL-2B runs on the inference worker; any OpenAI-compatible endpoint can be configured as an alternative." />
      <WithReconstruction>{(r, reload) => <AnalysisBody r={r} reload={reload} />}</WithReconstruction>
    </>
  );
}

function AnalysisBody({ r, reload }: { r: Reconstruction; reload: () => Promise<void> }) {
  const sys = useApi(() => api.system(), []);
  const [provider, setProvider] = useState("qwen3vl_local");
  const [model, setModel] = useState("");
  const [mode, setMode] = useState<"structured" | "caption">("structured");
  const [error, setError] = useState<string | null>(null);
  const [manual, setManual] = useState(false);
  const refAnalyses = r.analyses.filter((a) => a.image_id === r.image.id);
  const otherAnalyses = r.analyses.filter((a) => a.image_id !== r.image.id);
  const prov = sys.data?.vlm_providers.find((p) => p.name === provider);

  const run = async () => {
    setError(null);
    try {
      await api.requestAnalysis(r.id, { provider, model: model || null, mode });
      await reload();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const runBoth = async () => {
    setError(null);
    try {
      for (const m of ["caption", "structured"] as const) await api.requestAnalysis(r.id, { provider, model: model || null, mode: m });
      await reload();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const busy = refAnalyses.some((a) => ["pending", "queued", "running"].includes(a.status));

  return (
    <div className="grid gap-6 xl:grid-cols-[380px_1fr]">
      <div className="space-y-6">
        <ImageFrame src={r.image.url} label={r.name} sub={`${r.image.width}×${r.image.height} · sha256 ${r.image.sha256.slice(0, 12)}…`} />
        <Card title="Run analysis">
          <div className="space-y-3">
            <Field label="Provider">
              <select className="input" value={provider} onChange={(e) => { setProvider(e.target.value); setModel(""); }}>
                {sys.data?.vlm_providers.map((p) => <option key={p.name} value={p.name}>{p.name}{p.available ? "" : " (not configured)"}</option>)}
              </select>
            </Field>
            {prov && <p className="text-xs text-slate-400">{prov.description} <span className="text-slate-500">{prov.reason}</span></p>}
            {provider === "qwen3vl_local" ? (
              <Field label="Model (allow-listed)">
                <select className="input" value={model} onChange={(e) => setModel(e.target.value)}>
                  {Object.entries(sys.data?.local_vlm_models ?? {}).map(([id, label]) => <option key={id} value={id}>{label}</option>)}
                </select>
              </Field>
            ) : (
              <Field label="Model override" hint="Leave empty to use the configured default.">
                <input className="input" value={model} onChange={(e) => setModel(e.target.value)} placeholder={prov?.model} />
              </Field>
            )}
            <Field label="Mode">
              <div className="grid grid-cols-2 gap-2">
                {(["structured", "caption"] as const).map((m) => (
                  <button key={m} onClick={() => setMode(m)} className={mode === m ? "btn-primary" : "btn-secondary"}>{m === "structured" ? "Structured JSON" : "Caption (baseline A)"}</button>
                ))}
              </div>
            </Field>
            <ErrorText error={error} />
            <button className="btn-primary w-full" disabled={!prov?.available || busy} onClick={runBoth} title="Queue a caption analysis (condition A) and a structured analysis (conditions B and D)">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}{busy ? "Analysing…" : "Analyse image (caption + structured)"}
            </button>
            <button className="btn-secondary w-full" disabled={!prov?.available} onClick={run}><Play className="h-4 w-4" />Run only {mode} analysis</button>
            {prov?.execution === "worker" && !sys.data?.gpu_worker_online && <Notice tone="warn">No inference worker online — the job will queue until one connects.</Notice>}
          </div>
        </Card>
        <Card title="Manual annotation" subtitle="Human-written attributes, stored as provider “manual” — never confused with model output."
          actions={<button className="btn-secondary" onClick={() => setManual((v) => !v)}><PenLine className="h-4 w-4" />{manual ? "Close" : "Annotate"}</button>}>
          {manual ? <ManualForm r={r} onDone={async () => { setManual(false); await reload(); }} /> : <p className="text-sm text-slate-400">Useful for ablations or when no VLM is available.</p>}
        </Card>
      </div>

      <div className="space-y-4">
        {refAnalyses.length === 0 && <Empty title="No analyses yet">Run a structured analysis (for conditions B and C) and a caption analysis (for baseline A).</Empty>}
        {[...refAnalyses].reverse().map((a) => <AnalysisCard key={a.id} a={a} />)}
        {otherAnalyses.length > 0 && (
          <Card title="Analyses of generated images" subtitle="Used by the refinement proposer to detect discrepancies.">
            <div className="space-y-3">{[...otherAnalyses].reverse().map((a) => <AnalysisCard key={a.id} a={a} compact />)}</div>
          </Card>
        )}
      </div>
    </div>
  );
}

function AnalysisCard({ a, compact }: { a: Analysis; compact?: boolean }) {
  const [raw, setRaw] = useState(false);
  const { search } = useLocation();
  return (
    <Card pad
      title={<span className="flex flex-wrap items-center gap-2">{a.mode === "structured" ? "Structured analysis" : "Caption"}<StatusBadge status={a.status} />{a.is_manual ? <Pill tone="amber">human annotation</Pill> : <Pill tone="cyan">model output</Pill>}</span>}
      subtitle={<span className="mono text-xs">{a.provider} · {a.model}</span>}
      actions={a.status === "completed" && !compact && <Link to={{ pathname: "/prompts", search }} className="btn-secondary"><Wand2 className="h-4 w-4" />Build prompts</Link>}>
      {a.status === "completed" && a.result && <AttributeView a={a.result} />}
      {a.status === "completed" && a.caption && <p className="rounded-lg border border-ink-700 bg-ink-850 p-3 text-slate-100">“{a.caption}”</p>}
      {["queued", "pending", "running"].includes(a.status) && <p className="text-sm text-slate-400">Waiting for {a.provider === "qwen3vl_local" ? "the inference worker" : "the provider"}… no output is shown until the model actually responds.</p>}
      {a.error && <Notice tone="error">{a.error}</Notice>}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
        <KV items={[["Duration", fmtDuration(a.duration_s)], ["Created", fmtTime(a.created_at)],
          ...(a.runtime_info?.device ? [["Device", `${String(a.runtime_info.device)} · ${String(a.runtime_info.dtype ?? "")}`] as [string, string]] : []),
          ...(a.runtime_info?.generated_tokens !== undefined ? [["Tokens in/out", `${String(a.runtime_info.input_tokens)} / ${String(a.runtime_info.generated_tokens)}`] as [string, string]] : []),
          ...((a.runtime_info?.memory as { peak_process_gb?: number } | undefined)?.peak_process_gb !== undefined ? [["Peak worker memory", `${(a.runtime_info!.memory as { peak_process_gb: number }).peak_process_gb} GB`] as [string, string]] : [])]} />
        {a.raw_output && <button className="btn-ghost px-2 py-1 text-xs" onClick={() => setRaw((v) => !v)}>Raw model output <ChevronDown className={`h-3.5 w-3.5 transition ${raw ? "rotate-180" : ""}`} /></button>}
      </div>
      {raw && a.raw_output && <pre className="mono mt-2 max-h-80 overflow-auto whitespace-pre-wrap rounded-lg border border-ink-700 bg-ink-950 p-3 text-slate-300">{a.raw_output}</pre>}
    </Card>
  );
}

function ManualForm({ r, onDone }: { r: Reconstruction; onDone: () => Promise<void> }) {
  const [attrs, setAttrs] = useState<Attributes>(emptyAttributes());
  const [caption, setCaption] = useState("");
  const [annotator, setAnnotator] = useState("");
  const [error, setError] = useState<string | null>(null);
  const save = async (kind: "structured" | "caption") => {
    setError(null);
    try {
      await api.manualAnalysis(r.id, kind === "structured" ? { attributes: attrs, annotator } : { caption, annotator });
      await onDone();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  return (
    <div className="space-y-4">
      <Field label="Annotator"><input className="input" value={annotator} onChange={(e) => setAnnotator(e.target.value)} placeholder="initials" /></Field>
      <AttributeEditor value={attrs} onChange={setAttrs} />
      <button className="btn-primary w-full" disabled={!attrs.subject} onClick={() => save("structured")}>Save structured annotation</button>
      <div className="border-t border-ink-700 pt-4">
        <Field label="Caption"><input className="input" value={caption} onChange={(e) => setCaption(e.target.value)} /></Field>
        <button className="btn-secondary mt-2 w-full" disabled={!caption} onClick={() => save("caption")}>Save caption</button>
      </div>
      <ErrorText error={error} />
    </div>
  );
}
