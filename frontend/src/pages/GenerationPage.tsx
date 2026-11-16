import { Columns3, Play } from "lucide-react";
import { useEffect, useState } from "react";
import { defaultParams, GenParamsForm } from "../components/GenParamsForm";
import { versionLabel } from "../components/PromptBits";
import { Card, Empty, ErrorText, Field, ImageFrame, KV, Notice, Pill, StatusBadge } from "../components/ui";
import { WithReconstruction } from "../components/WithReconstruction";
import { useApi } from "../hooks/useApi";
import { PageHeader } from "../layouts/AppLayout";
import { CONDITION_LABEL, fmtDuration, fmtTime } from "../lib/format";
import { api } from "../services/api";
import type { GenParams, Generation, Reconstruction } from "../types/api";

export function GenerationPage() {
  return (
    <>
      <PageHeader stage="Stage 3" title="Image generation" description="Generate from any prompt version with explicit, recorded parameters. Local diffusers jobs on the inference worker are fully seeded; hosted providers are flagged as not reproducible." />
      <WithReconstruction>{(r, reload) => <GenBody r={r} reload={reload} />}</WithReconstruction>
    </>
  );
}

function GenBody({ r, reload }: { r: Reconstruction; reload: () => Promise<void> }) {
  const sys = useApi(() => api.system(), []);
  const models = sys.data?.models ?? [];
  const [params, setParams] = useState<GenParams>(defaultParams());
  const [provider, setProvider] = useState("gpu_worker");
  const [pvId, setPvId] = useState(r.prompt_versions.at(-1)?.id ?? "");
  const [condition, setCondition] = useState("");
  const [error, setError] = useState<string | null>(null);
  const captions = r.prompt_versions.filter((p) => p.source === "caption");
  const detailed = r.prompt_versions.filter((p) => p.source !== "caption" && p.template_id === "natural_v1");
  const structured = r.prompt_versions.filter((p) => p.source !== "caption" && p.template_id !== "natural_v1");
  const [capId, setCapId] = useState(captions.at(-1)?.id ?? "");
  const [detId, setDetId] = useState(detailed.at(-1)?.id ?? "");
  const [strId, setStrId] = useState(structured.filter((p) => p.source === "validated").at(-1)?.id ?? structured.at(-1)?.id ?? "");
  useEffect(() => { if (!detId && detailed.length) setDetId(detailed.at(-1)!.id); }, [detailed, detId]);
  useEffect(() => { if (!capId && captions.length) setCapId(captions.at(-1)!.id); }, [captions, capId]);
  const latestStructured = [...r.analyses].reverse().find((a) => a.mode === "structured" && a.status === "completed" && a.image_id === r.image.id);
  const makeDetailed = async () => {
    if (!latestStructured) return;
    setError(null);
    try {
      const pvs = await api.promptsFromAnalysis(r.id, latestStructured.id, "natural_v1");
      setDetId(pvs.at(-1)!.id);
      await reload();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };

  useEffect(() => { if (models.length && !models.find((m) => m.key === params.model_key)) setParams(defaultParams(models[0])); }, [models, params.model_key]);
  useEffect(() => { if (!pvId && r.prompt_versions.length) setPvId(r.prompt_versions.at(-1)!.id); }, [r.prompt_versions, pvId]);

  const run = async () => {
    setError(null);
    try { await api.generate(r.id, { prompt_version_id: pvId, provider, params, condition: condition || null }); await reload(); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const compare = async () => {
    setError(null);
    try {
      await api.startComparison(r.id, { caption_prompt_version_id: capId || undefined, detailed_prompt_version_id: detId || undefined, structured_prompt_version_id: strId || undefined, provider, params: { ...params, negative_prompt: "" } });
      await reload();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const prov = sys.data?.generation_providers.find((p) => p.name === provider);

  return (
    <div className="grid gap-6 xl:grid-cols-[400px_1fr]">
      <div className="space-y-6">
        <Card title="Configuration">
          {r.prompt_versions.length === 0 ? <Empty title="No prompt versions">Create prompts in Stage 2 first.</Empty> : (
            <div className="space-y-3">
              <Field label="Provider">
                <select className="input" value={provider} onChange={(e) => setProvider(e.target.value)}>
                  {sys.data?.generation_providers.map((p) => <option key={p.name} value={p.name}>{p.name}{p.available ? "" : " (not configured)"}</option>)}
                </select>
              </Field>
              {prov && <p className="text-xs text-slate-400">{prov.description}</p>}
              <GenParamsForm models={models} value={params} onChange={setParams} />
            </div>
          )}
        </Card>
        {r.prompt_versions.length > 0 && (
          <>
            <Card title="Single generation">
              <div className="space-y-3">
                <Field label="Prompt version">
                  <select className="input" value={pvId} onChange={(e) => setPvId(e.target.value)}>
                    {r.prompt_versions.map((p) => <option key={p.id} value={p.id}>{versionLabel(p)}</option>)}
                  </select>
                </Field>
                <Field label="Condition label (optional)">
                  <select className="input" value={condition} onChange={(e) => setCondition(e.target.value)}>
                    <option value="">None</option>
                    {Object.entries(CONDITION_LABEL).filter(([k]) => k !== "C_refined").map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                  </select>
                </Field>
                <button className="btn-primary w-full" disabled={!pvId || !prov?.available} onClick={run}><Play className="h-4 w-4" />Generate</button>
              </div>
            </Card>
            <Card title="Controlled comparison (A · D · B)" subtitle="Same model, seed, steps, CFG and resolution — only the prompt differs.">
              <div className="space-y-3">
                <Field label="A · Caption prompt">
                  <select className="input" value={capId} onChange={(e) => setCapId(e.target.value)}>
                    <option value="">— none —</option>{captions.map((p) => <option key={p.id} value={p.id}>{versionLabel(p)}</option>)}
                  </select>
                </Field>
                <Field label="D · Detailed prose prompt" hint="natural_v1 template over the same structured analysis">
                  <div className="flex gap-1.5">
                    <select className="input" value={detId} onChange={(e) => setDetId(e.target.value)}>
                      <option value="">— none —</option>{detailed.map((p) => <option key={p.id} value={p.id}>{versionLabel(p)}</option>)}
                    </select>
                    {latestStructured && <button className="btn-secondary shrink-0 px-2.5 text-xs" onClick={makeDetailed} title="Build a detailed prose prompt (natural_v1) from the latest structured analysis">Create</button>}
                  </div>
                </Field>
                <Field label="B · Structured prompt">
                  <select className="input" value={strId} onChange={(e) => setStrId(e.target.value)}>
                    <option value="">— none —</option>{structured.map((p) => <option key={p.id} value={p.id}>{versionLabel(p)}</option>)}
                  </select>
                </Field>
                <p className="text-xs text-slate-400">Condition C is produced by a refinement session started from B (Stage 5).</p>
                <button className="btn-secondary w-full" disabled={(!capId && !strId && !detId) || !prov?.available} onClick={compare}><Columns3 className="h-4 w-4" />Run selected conditions with identical parameters</button>
              </div>
            </Card>
          </>
        )}
        <ErrorText error={error} />
        {prov?.execution === "worker" && !sys.data?.gpu_worker_online && <Notice tone="warn">No inference worker online. Jobs will be queued and executed when a worker connects (scripts/dev.sh starts one on this Mac).</Notice>}
      </div>

      <div className="grid content-start gap-4 md:grid-cols-2 2xl:grid-cols-3">
        {r.generations.length === 0 && <div className="md:col-span-2 2xl:col-span-3"><Empty title="No generations yet" /></div>}
        {[...r.generations].reverse().map((g) => <GenerationCard key={g.id} g={g} />)}
      </div>
    </div>
  );
}

export function GenerationCard({ g }: { g: Generation }) {
  return (
    <div className="card overflow-hidden">
      <ImageFrame className="rounded-none border-0" src={g.output_image?.url}
        placeholder={g.status === "failed" ? "Generation failed" : g.provider === "gpu_worker" ? `${g.status} — awaiting worker` : g.status}
        label={<span className="flex items-center gap-2">v{g.prompt_version} · {g.model_key}<StatusBadge status={g.status} /></span>}
        sub={g.condition ? <Pill tone="accent">{CONDITION_LABEL[g.condition]}</Pill> : undefined} />
      <div className="space-y-2 p-3 text-xs">
        <p className="mono line-clamp-3 text-slate-300" title={g.positive}>{g.positive}</p>
        <KV items={[["seed", <span className="font-mono">{g.params.seed}</span>], ["steps / cfg", <span className="font-mono">{g.params.steps} / {g.params.cfg_scale}</span>],
          ["size", <span className="font-mono">{g.params.width}×{g.params.height}</span>], ["provider", g.provider], ["time", fmtDuration(g.duration_s)],
          ...(g.runtime_info?.device ? [["device", `${String(g.runtime_info.device)} · ${String(g.runtime_info.dtype ?? "")}`] as [string, string]] : []),
          ...((g.runtime_info?.memory as { peak_process_gb?: number } | undefined)?.peak_process_gb !== undefined ? [["peak memory", `${(g.runtime_info!.memory as { peak_process_gb: number }).peak_process_gb} GB`] as [string, string]] : []),
          ["created", fmtTime(g.created_at)]]} />
        {g.runtime_info && g.runtime_info.reproducible === false && <Pill tone="amber">remote · not reproducible</Pill>}
        {g.error && <p className="text-rose-300">{g.error}</p>}
      </div>
    </div>
  );
}
