import { Download, FileJson, FilePlus2, Fingerprint, PackageOpen } from "lucide-react";
import { useState } from "react";
import { Card, Empty, ErrorText, Field, KV, Notice } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { useActiveReconstruction } from "../hooks/useActiveReconstruction";
import { PageHeader } from "../layouts/AppLayout";
import { fmtTime } from "../lib/format";
import { api } from "../services/api";

export function ExperimentsPage() {
  const exps = useApi(() => api.experiments(), []);
  const reports = useApi(() => api.reports(), []);
  const bundles = useApi(() => api.demoBundles(), []);
  const fp = useApi(() => api.fingerprint(), []);
  const { activeId } = useActiveReconstruction();
  const [name, setName] = useState("");
  const [hypothesis, setHypothesis] = useState("");
  const [description, setDescription] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (fn: () => Promise<unknown>, ok: string) => {
    setError(null); setMsg(null);
    try { await fn(); setMsg(ok); await Promise.all([exps.reload(), reports.reload(), bundles.reload()]); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };

  return (
    <>
      <PageHeader title="Experiments" description="Every run records the reference image hash, prompt versions, model versions, seed, parameters, metrics, iteration, timing and a code fingerprint. Reports are written to experiments/reports/ with timestamped names and are never overwritten." />
      {msg && <div className="mb-4"><Notice tone="ok">{msg}</Notice></div>}
      <div className="mb-4"><ErrorText error={error} /></div>
      <div className="grid gap-6 xl:grid-cols-[360px_1fr]">
        <div className="space-y-6">
          <Card title="New experiment">
            <div className="space-y-3">
              <Field label="Name"><input className="input" value={name} onChange={(e) => setName(e.target.value)} /></Field>
              <Field label="Hypothesis"><textarea className="input min-h-16" value={hypothesis} onChange={(e) => setHypothesis(e.target.value)} placeholder="e.g. Structured prompts (B) yield higher DINOv2 similarity than captions (A) at equal seed." /></Field>
              <Field label="Description"><textarea className="input min-h-16" value={description} onChange={(e) => setDescription(e.target.value)} /></Field>
              <button className="btn-primary w-full" disabled={!name} onClick={() => run(() => api.createExperiment({ name, hypothesis, description }).then(() => { setName(""); setHypothesis(""); setDescription(""); }), "Experiment created. Attach reconstructions when uploading.")}><FilePlus2 className="h-4 w-4" />Create</button>
            </div>
          </Card>
          <Card title="Code fingerprint" actions={<Fingerprint className="h-4 w-4 text-slate-500" />}>
            {fp.data && <KV items={[["git commit", <span className="mono">{String(fp.data.git_commit ?? "—").slice(0, 12)}</span>], ["dirty", String(fp.data.git_dirty)],
              ["source sha256", <span className="mono">{String(fp.data.source_sha256).slice(0, 16)}…</span>], ["python", String(fp.data.python)]]} />}
          </Card>
          <Card title="Demo bundles" subtitle="Portable A/B/C comparisons for presentation (experiments/demo/)">
            {activeId && <button className="btn-secondary mb-3 w-full" onClick={() => run(() => api.exportDemoBundle(activeId), "Demo bundle exported from the active reconstruction.")}><PackageOpen className="h-4 w-4" />Export active reconstruction</button>}
            {bundles.data?.length ? <ul className="space-y-1 text-sm">{bundles.data.map((b) => <li key={b.id} className="truncate text-slate-300" title={b.id}>{b.name} <span className="text-slate-500">· {fmtTime(b.created_at)}</span></li>)}</ul>
              : <p className="text-sm text-slate-500">None yet.</p>}
          </Card>
        </div>
        <div className="space-y-6">
          {exps.data?.length ? exps.data.map((e) => (
            <Card key={e.id} title={e.name} subtitle={`${e.reconstructions.length} reconstruction(s) · created ${fmtTime(e.created_at)}`}
              actions={<>
                <a className="btn-secondary" href={`/api/experiments/${e.id}/export.json`}><FileJson className="h-4 w-4" />JSON</a>
                <a className="btn-secondary" href={`/api/experiments/${e.id}/export.csv`}><Download className="h-4 w-4" />CSV</a>
                <button className="btn-primary" onClick={() => run(() => api.writeReport(e.id), "Report written to experiments/reports/.")}>Write report</button>
              </>}>
              {e.hypothesis && <p className="mb-3 text-sm text-slate-300"><span className="label mr-2">Hypothesis</span>{e.hypothesis}</p>}
              {e.reconstructions.length ? (
                <div className="flex flex-wrap gap-2">{e.reconstructions.map((r) => (
                  <div key={r.id} className="flex items-center gap-2 rounded-md border border-ink-700 bg-ink-850 p-1.5 pr-3 text-xs">
                    <img src={r.image.url} alt="" className="h-8 w-8 rounded object-cover" /><span>{r.name}<span className="block text-slate-500">{r.n_completed_generations} images</span></span>
                  </div>))}</div>
              ) : <p className="text-sm text-slate-500">No reconstructions attached.</p>}
            </Card>
          )) : <Empty title="No experiments yet" />}
          <Card title="Written reports" pad={false}>
            {reports.data?.length ? (
              <ul className="divide-y divide-ink-800 text-sm">{reports.data.map((r) => (
                <li key={r.name} className="flex items-center justify-between px-5 py-2"><a className="mono text-accent-300 hover:underline" href={`/api/reports/${r.name}`}>{r.name}</a><span className="text-xs text-slate-500">{(r.size_bytes / 1024).toFixed(1)} KB</span></li>
              ))}</ul>
            ) : <div className="p-5"><p className="text-sm text-slate-500">No reports written yet.</p></div>}
          </Card>
        </div>
      </div>
    </>
  );
}
