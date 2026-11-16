import { GitCompare, Save, Sparkles, Wand2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { AttributeEditor } from "../components/AttributeEditor";
import { PromptDiffView, PromptText, SegmentedPrompt, SourcePill, TokenMeter, versionLabel } from "../components/PromptBits";
import { Card, Empty, ErrorText, Field, IssueList, Notice } from "../components/ui";
import { WithReconstruction } from "../components/WithReconstruction";
import { useApi } from "../hooks/useApi";
import { PageHeader } from "../layouts/AppLayout";
import { fmtTime } from "../lib/format";
import { api } from "../services/api";
import type { Attributes, BuiltPrompt, Reconstruction } from "../types/api";

export function PromptStudioPage() {
  return (
    <>
      <PageHeader stage="Stage 2" title="Prompt studio" description="Fixed, versioned templates turn attributes into positive and negative prompts. Every edit creates a new immutable version; validation flags contradictions, duplicates and inconsistent relationships." />
      <WithReconstruction>{(r, reload) => <Studio r={r} reload={reload} />}</WithReconstruction>
    </>
  );
}

function Studio({ r, reload }: { r: Reconstruction; reload: () => Promise<void> }) {
  const sys = useApi(() => api.system(), []);
  const { search } = useLocation();
  const versions = r.prompt_versions;
  const [selectedId, setSelectedId] = useState<string | null>(versions.at(-1)?.id ?? null);
  const selected = versions.find((v) => v.id === selectedId) ?? versions.at(-1) ?? null;
  const completed = r.analyses.filter((a) => a.status === "completed" && a.image_id === r.image.id);
  const [analysisId, setAnalysisId] = useState(completed.at(-1)?.id ?? "");
  const [template, setTemplate] = useState("structured_v1");
  const [error, setError] = useState<string | null>(null);
  const [diffA, setDiffA] = useState("");
  const [diffB, setDiffB] = useState("");

  useEffect(() => { if (!analysisId && completed.length) setAnalysisId(completed.at(-1)!.id); }, [completed, analysisId]);

  const build = async () => {
    setError(null);
    try {
      const out = await api.promptsFromAnalysis(r.id, analysisId, template);
      await reload();
      setSelectedId(out.at(-1)!.id);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };

  return (
    <div className="grid gap-6 xl:grid-cols-[320px_1fr]">
      <div className="space-y-6">
        <Card title="Build from analysis">
          {completed.length ? (
            <div className="space-y-3">
              <Field label="Analysis">
                <select className="input" value={analysisId} onChange={(e) => setAnalysisId(e.target.value)}>
                  {completed.map((a) => <option key={a.id} value={a.id}>{a.mode} · {a.provider} · {fmtTime(a.created_at)}</option>)}
                </select>
              </Field>
              {completed.find((a) => a.id === analysisId)?.mode === "structured" && (
                <Field label="Template">
                  <select className="input" value={template} onChange={(e) => setTemplate(e.target.value)}>
                    {sys.data?.templates.filter((t) => t.id !== "caption_v1").map((t) => <option key={t.id} value={t.id}>{t.id}</option>)}
                  </select>
                </Field>
              )}
              <p className="text-xs text-slate-400">{sys.data?.templates.find((t) => t.id === template)?.description}</p>
              <ErrorText error={error} />
              <button className="btn-primary w-full" onClick={build} disabled={!analysisId}><Wand2 className="h-4 w-4" />Generate prompt versions</button>
            </div>
          ) : <Empty title="No completed analysis">Run Stage 1 first, or add a manual annotation.</Empty>}
        </Card>
        <Card title="Versions" pad={false}>
          <ul className="max-h-[480px] divide-y divide-ink-800 overflow-y-auto">
            {[...versions].reverse().map((v) => (
              <li key={v.id}>
                <button onClick={() => setSelectedId(v.id)} className={`w-full px-4 py-2.5 text-left transition ${selected?.id === v.id ? "bg-accent-500/10" : "hover:bg-ink-800"}`}>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm text-slate-100">v{v.version}</span><SourcePill source={v.source} />
                    {v.validation && (v.validation.ok ? <span className="ml-auto text-xs text-emerald-400">valid</span>
                      : <span className="ml-auto text-xs text-amber-300">{v.validation.issues.filter((i) => i.severity === "error").length} error(s)</span>)}
                  </div>
                  <p className="mt-0.5 truncate text-xs text-slate-400">{v.note || v.template_id}</p>
                </button>
              </li>
            ))}
            {!versions.length && <li className="px-4 py-6 text-center text-sm text-slate-500">No versions yet.</li>}
          </ul>
        </Card>
      </div>

      <div className="space-y-6">
        {selected ? (
          <>
            <Card title={<span className="flex items-center gap-2">Version {selected.version} <SourcePill source={selected.source} /></span>}
              subtitle={`${selected.template_id} · ${selected.note}`}
              actions={<Link to={{ pathname: "/generation", search }} className="btn-secondary"><Sparkles className="h-4 w-4" />Generate with this</Link>}>
              <div className="space-y-4">
                {selected.validation?.segments && selected.source !== "caption" && <SegmentedPrompt segments={selected.validation.segments} />}
                <PromptText label="Positive prompt" text={selected.positive} />
                <PromptText label="Negative prompt" text={selected.negative} />
                {selected.validation && <TokenMeter v={selected.validation} />}
                {selected.validation?.fixes_applied?.length ? (
                  <Notice tone="ok" title="Automatic fixes applied (original preserved as parent version)">
                    <ul className="list-disc pl-4">{selected.validation.fixes_applied.map((f, i) => <li key={i}>{f}</li>)}</ul>
                  </Notice>
                ) : null}
                {selected.validation && <div><p className="label mb-1.5">Validation</p><IssueList issues={selected.validation.issues} /></div>}
              </div>
            </Card>
            {selected.source !== "caption" && <Editor key={selected.id} r={r} parentId={selected.id} initial={selected.attributes} template={selected.template_id}
              onSaved={async (id) => { await reload(); setSelectedId(id); }} />}
          </>
        ) : <Empty title="No prompt selected">Generate prompt versions from an analysis.</Empty>}

        {versions.length >= 2 && (
          <Card title="Compare versions" actions={<GitCompare className="h-4 w-4 text-slate-500" />}>
            <div className="mb-4 grid grid-cols-2 gap-3">
              {[[diffA, setDiffA, "From"], [diffB, setDiffB, "To"]].map(([val, set, label]) => (
                <Field key={label as string} label={label as string}>
                  <select className="input" value={val as string} onChange={(e) => (set as (s: string) => void)(e.target.value)}>
                    <option value="">Select…</option>
                    {versions.map((v) => <option key={v.id} value={v.id}>{versionLabel(v)}</option>)}
                  </select>
                </Field>
              ))}
            </div>
            {diffA && diffB && diffA !== diffB ? <PromptDiffView a={diffA} b={diffB} /> : <p className="text-sm text-slate-500">Choose two different versions.</p>}
          </Card>
        )}
      </div>
    </div>
  );
}

function Editor({ r, parentId, initial, template: initialTemplate, onSaved }: {
  r: Reconstruction; parentId: string; initial: Attributes; template: string; onSaved: (id: string) => Promise<void>;
}) {
  const sys = useApi(() => api.system(), []);
  const [attrs, setAttrs] = useState<Attributes>(initial);
  const [template, setTemplate] = useState(initialTemplate === "caption_v1" ? "structured_v1" : initialTemplate);
  const [note, setNote] = useState("");
  const [preview, setPreview] = useState<{ original: BuiltPrompt; validated: BuiltPrompt } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dirty = useMemo(() => JSON.stringify(attrs) !== JSON.stringify(initial) || template !== initialTemplate, [attrs, initial, template, initialTemplate]);

  useEffect(() => {
    const t = setTimeout(() => {
      api.previewPrompt(attrs, template).then(setPreview).catch((e) => setError(e.message));
    }, 300);
    return () => clearTimeout(t);
  }, [attrs, template]);

  const save = async () => {
    setError(null);
    try {
      const pv = await api.createPromptVersion(r.id, { parent_id: parentId, attributes: attrs, template_id: template, note: note || "Manual edit" });
      await onSaved(pv.id);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };

  return (
    <Card title="Edit attributes → new version" subtitle="Live preview re-runs the template and validator on every change. Saving never modifies the parent version.">
      <div className="grid gap-6 2xl:grid-cols-2">
        <div className="space-y-4">
          <Field label="Template">
            <select className="input" value={template} onChange={(e) => setTemplate(e.target.value)}>
              {sys.data?.templates.filter((t) => t.id !== "caption_v1").map((t) => <option key={t.id} value={t.id}>{t.id}</option>)}
            </select>
          </Field>
          <AttributeEditor value={attrs} onChange={setAttrs} />
        </div>
        <div className="space-y-4">
          {preview && (
            <>
              <PromptText label="Preview · positive" text={preview.original.positive} />
              <PromptText label="Preview · negative" text={preview.original.negative} />
              <TokenMeter v={preview.original.validation} />
              <IssueList issues={preview.original.validation.issues} />
            </>
          )}
          <Field label="Change note"><input className="input" value={note} onChange={(e) => setNote(e.target.value)} placeholder="What and why" /></Field>
          <ErrorText error={error} />
          <button className="btn-primary w-full" disabled={!dirty} onClick={save}><Save className="h-4 w-4" />Save as new version</button>
        </div>
      </div>
    </Card>
  );
}
