import clsx from "clsx";
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import { useActiveReconstruction } from "../hooks/useActiveReconstruction";
import { CONDITION_COLOR } from "../lib/chart";
import { CONDITION_LABEL, fmtDuration, fmtNum } from "../lib/format";
import { api } from "../services/api";
import type { Comparison, DemoBundle, GenParams } from "../types/api";

type CondKey = "A_caption" | "B_structured" | "C_refined";
const CONDS: CondKey[] = ["A_caption", "B_structured", "C_refined"];

interface ViewCond {
  status: string; imageUrl?: string; prompt?: string; negative?: string; modelId?: string; provider?: string;
  params?: GenParams; duration?: number | null; promptVersion?: number; promptSource?: string;
  metrics: { label: string; metric: string; value: number | null; status: string; role: string; hib: boolean }[];
}
interface View { name: string; referenceUrl: string; source: string; conds: Record<CondKey, ViewCond>;
  refinement?: { best_iteration: number; optimization_metric: string; history: { iteration: number; status: string; score: number | null }[] } | null }

const METRIC_LABEL: Record<string, string> = { clip_l14: "CLIP L/14", clip_b32: "CLIP B/32", dinov2_base: "DINOv2", ssim: "SSIM", lpips_alex: "LPIPS" };

function fromLive(c: Comparison): View {
  const conds = {} as Record<CondKey, ViewCond>;
  for (const k of CONDS) {
    const g = c.conditions[k];
    conds[k] = g ? {
      status: g.status, imageUrl: g.output_image?.url, prompt: g.positive, negative: g.params.negative_prompt, modelId: g.model_id,
      provider: g.provider, params: g.params, duration: g.duration_s, promptVersion: g.prompt_version,
      metrics: g.metrics.map((m) => ({ label: METRIC_LABEL[m.metric] ?? m.metric, metric: m.metric, value: m.value, status: m.status, role: m.role, hib: m.higher_is_better })),
    } : { status: "not_executed", metrics: [] };
  }
  return { name: c.reconstruction.name, referenceUrl: c.reconstruction.image.url, source: "Live database", conds, refinement: c.refinement };
}

function fromBundle(id: string, b: DemoBundle): View {
  const conds = {} as Record<CondKey, ViewCond>;
  for (const k of CONDS) {
    const d = b.conditions[k];
    conds[k] = {
      status: d?.status ?? "not_executed", imageUrl: d?.image ? api.demoFileUrl(id, d.image) : undefined, prompt: d?.prompt,
      negative: d?.negative_prompt, modelId: d?.model_id, provider: d?.provider, params: d?.params, duration: d?.duration_s,
      promptVersion: d?.prompt_version, promptSource: d?.prompt_source,
      metrics: (d?.metrics ?? []).map((m) => ({ label: METRIC_LABEL[m.metric] ?? m.metric, metric: m.metric, value: m.value, status: m.status, role: m.role, hib: m.higher_is_better })),
    };
  }
  return { name: b.name, referenceUrl: api.demoFileUrl(id, "reference.jpg"), source: `Demo bundle · ${id}`, conds, refinement: b.conditions.C_refined?.refinement ?? null };
}

export function PresentationPage() {
  const navigate = useNavigate();
  const { search } = useLocation();
  const { activeId } = useActiveReconstruction();
  const bundles = useApi(() => api.demoBundles(), []);
  const recons = useApi(() => api.reconstructions(), []);
  const [source, setSource] = useState<string>(activeId ? `live:${activeId}` : "");
  useEffect(() => { if (!source && bundles.data?.length) setSource(`bundle:${bundles.data[0].id}`); }, [bundles.data, source]);
  const [kind, id] = source.split(":") as ["live" | "bundle" | "", string];
  const live = useApi(kind === "live" ? () => api.comparison(id) : null, [source], { pollMs: 5000 });
  const bundle = useApi(kind === "bundle" ? () => api.demoBundle(id) : null, [source]);
  const view = useMemo<View | null>(() => (kind === "live" && live.data ? fromLive(live.data) : kind === "bundle" && bundle.data ? fromBundle(id, bundle.data) : null), [kind, id, live.data, bundle.data]);

  const [slide, setSlide] = useState(0);
  const slides = buildSlides(view);
  const go = useCallback((d: number) => setSlide((s) => Math.max(0, Math.min(slides.length - 1, s + d))), [slides.length]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "SELECT") return;
      if (["ArrowRight", "PageDown", " "].includes(e.key)) { e.preventDefault(); go(1); }
      if (["ArrowLeft", "PageUp"].includes(e.key)) go(-1);
      if (e.key === "Escape") navigate({ pathname: "/", search });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, navigate, search]);

  return (
    <div className="relative flex min-h-screen flex-col bg-[radial-gradient(ellipse_at_top,#151b3a_0%,#070a10_55%)] text-slate-100">
      <header className="flex items-center gap-3 px-8 py-4 text-sm text-slate-400">
        <span className="font-semibold text-slate-200">Image-to-Prompt Reconstruction</span>
        <select className="input ml-auto w-auto max-w-sm py-1.5" value={source} onChange={(e) => { setSource(e.target.value); setSlide(0); }}>
          <option value="">Choose data source…</option>
          <optgroup label="Demo bundles (precomputed, genuine)">{bundles.data?.map((b) => <option key={b.id} value={`bundle:${b.id}`}>{b.name} · {b.id.slice(0, 15)}</option>)}</optgroup>
          <optgroup label="Live reconstructions">{recons.data?.map((r) => <option key={r.id} value={`live:${r.id}`}>{r.name}</option>)}</optgroup>
        </select>
        <button className="btn-ghost" onClick={() => navigate({ pathname: "/", search })} title="Exit (Esc)"><X className="h-5 w-5" /></button>
      </header>
      <main className="flex flex-1 items-center justify-center px-10 pb-6">{slides[slide]}</main>
      <footer className="flex items-center gap-4 px-8 pb-5 text-sm text-slate-500">
        <button className="btn-ghost" onClick={() => go(-1)} disabled={slide === 0}><ChevronLeft className="h-5 w-5" /></button>
        <div className="flex flex-1 justify-center gap-1.5">{slides.map((_, i) => <button key={i} onClick={() => setSlide(i)} className={clsx("h-1.5 rounded-full transition-all", i === slide ? "w-8 bg-accent-400" : "w-3 bg-ink-600")} />)}</div>
        <span>{view?.source ?? "no data source"}</span>
        <button className="btn-ghost" onClick={() => go(1)} disabled={slide === slides.length - 1}><ChevronRight className="h-5 w-5" /></button>
      </footer>
    </div>
  );
}

function Title({ kicker, children }: { kicker?: string; children: React.ReactNode }) {
  return <div className="mb-8">{kicker && <p className="mb-2 text-sm font-semibold uppercase tracking-[0.2em] text-accent-300">{kicker}</p>}<h2 className="text-4xl font-semibold tracking-tight text-white">{children}</h2></div>;
}

function NotRun({ status }: { status: string }) {
  const text = status === "not_executed" ? "Not executed" : status === "failed" ? "Failed" : `Pending · ${status}`;
  return <div className="flex h-full w-full flex-col items-center justify-center gap-2 border-2 border-dashed border-ink-600 text-center"><p className="text-xl font-medium text-slate-400">{text}</p><p className="max-w-[16rem] text-sm text-slate-500">No image is shown because this stage has not produced a genuine output.</p></div>;
}

function buildSlides(view: View | null): React.ReactNode[] {
  const intro = (
    <div className="max-w-5xl text-center">
      <p className="mb-4 text-sm font-semibold uppercase tracking-[0.25em] text-accent-300">Research demonstration</p>
      <h1 className="bg-gradient-to-br from-white to-slate-400 bg-clip-text text-6xl font-semibold tracking-tight text-transparent">Image-to-Prompt Reconstruction</h1>
      <p className="mx-auto mt-6 max-w-3xl text-2xl leading-relaxed text-slate-300">Can a vision-language model describe an image well enough that a diffusion model regenerates it?</p>
      <p className="mt-10 text-base text-slate-500">→ to advance · Esc to exit</p>
    </div>
  );
  const pipeline = (
    <div className="w-full max-w-6xl">
      <Title kicker="Method">Five-stage pipeline</Title>
      <div className="grid grid-cols-5 gap-4">
        {[["Analyse", "Qwen3-VL-2B → structured JSON"], ["Prompt", "Fixed templates + validation"], ["Generate", "SD 1.5 · SD-Turbo · SDXL, seeded"], ["Evaluate", "CLIP · DINOv2 · SSIM · LPIPS"], ["Refine", "Propose → approve → early stop"]].map(([t, d], i) => (
          <div key={t} className="rounded-2xl border border-ink-600 bg-ink-900/70 p-6">
            <p className="text-5xl font-semibold text-accent-400/80">{i + 1}</p><p className="mt-4 text-2xl font-semibold">{t}</p><p className="mt-2 text-lg text-slate-400">{d}</p>
          </div>
        ))}
      </div>
      <p className="mt-8 text-xl text-slate-400">Conditions compared at identical model, seed and parameters: <b className="text-slate-200">A</b> caption only · <b className="text-slate-200">B</b> structured prompt · <b className="text-slate-200">C</b> structured + refinement.</p>
    </div>
  );
  if (!view) return [intro, pipeline, <div key="none" className="text-center text-2xl text-slate-400">Choose a demo bundle or reconstruction (top right) to show results.</div>];

  const reference = (
    <div className="flex w-full max-w-6xl items-center gap-12">
      <img src={view.referenceUrl} alt="reference" className="max-h-[68vh] w-1/2 rounded-2xl border border-ink-600 object-contain" />
      <div><Title kicker="Reference image">{view.name}</Title><p className="text-2xl leading-relaxed text-slate-300">The only input. Everything that follows is derived from this image by models — no human-written prompt.</p></div>
    </div>
  );
  const compare = (
    <div className="w-full max-w-[1600px]">
      <Title kicker="Results">Original vs. three reconstructions</Title>
      <div className="grid grid-cols-4 gap-5">
        <figure><div className="aspect-square overflow-hidden rounded-xl border-2 border-slate-400/60 bg-ink-900"><img src={view.referenceUrl} alt="" className="h-full w-full object-contain" /></div><figcaption className="mt-3 text-xl font-semibold">Original</figcaption></figure>
        {CONDS.map((k) => {
          const c = view.conds[k];
          return (
            <figure key={k}>
              <div className="aspect-square overflow-hidden rounded-xl border-2 bg-ink-900" style={{ borderColor: CONDITION_COLOR[k] }}>
                {c.status === "completed" && c.imageUrl ? <img src={c.imageUrl} alt="" className="h-full w-full object-contain" /> : <NotRun status={c.status} />}
              </div>
              <figcaption className="mt-3 text-xl font-semibold">{CONDITION_LABEL[k]}</figcaption>
              {c.params && <p className="font-mono text-sm text-slate-400">{c.params.model_key} · seed {c.params.seed} · {c.params.steps} steps · cfg {c.params.cfg_scale}</p>}
            </figure>
          );
        })}
      </div>
    </div>
  );
  const prompts = (
    <div className="w-full max-w-[1600px]">
      <Title kicker="Actual prompts & configuration">What each condition sent to the model</Title>
      <div className="grid grid-cols-3 gap-5">
        {CONDS.map((k) => {
          const c = view.conds[k];
          return (
            <div key={k} className="rounded-xl border-t-4 bg-ink-900/80 p-5" style={{ borderTopColor: CONDITION_COLOR[k] }}>
              <p className="text-lg font-semibold">{CONDITION_LABEL[k]}</p>
              {c.prompt ? <>
                <p className="mt-3 max-h-52 overflow-auto font-mono text-[15px] leading-relaxed text-slate-200">{c.prompt}</p>
                <p className="mt-3 text-xs uppercase tracking-wider text-slate-500">Negative</p>
                <p className="line-clamp-3 font-mono text-sm text-slate-400">{c.negative || "—"}</p>
                <dl className="mt-4 grid grid-cols-2 gap-x-3 gap-y-1 text-sm">
                  {[["Model", c.modelId], ["Provider", c.provider], ["Seed", c.params?.seed], ["Steps / CFG", c.params ? `${c.params.steps} / ${c.params.cfg_scale}` : "—"], ["Size", c.params ? `${c.params.width}×${c.params.height}` : "—"], ["Time", fmtDuration(c.duration)]].map(([kk, v]) => (
                    <div key={kk as string} className="contents"><dt className="text-slate-500">{kk}</dt><dd className="truncate font-mono text-slate-300" title={String(v ?? "")}>{String(v ?? "—")}</dd></div>
                  ))}
                </dl>
              </> : <p className="mt-3 text-lg text-slate-500">Not executed.</p>}
            </div>
          );
        })}
      </div>
    </div>
  );
  const metricKeys = ["clip_l14", "dinov2_base", "ssim", "lpips_alex"];
  const metrics = (
    <div className="w-full max-w-6xl">
      <Title kicker="Independent evaluation">Similarity to the original</Title>
      <table className="w-full text-xl">
        <thead><tr className="border-b border-ink-600 text-left text-base text-slate-400"><th className="py-3 font-medium">Condition</th>{metricKeys.map((m) => <th key={m} className="py-3 text-right font-medium">{METRIC_LABEL[m]} {m === "lpips_alex" ? "↓" : "↑"}</th>)}</tr></thead>
        <tbody>
          {CONDS.map((k) => (
            <tr key={k} className="border-b border-ink-800">
              <td className="py-4"><span className="mr-3 inline-block h-3 w-3 rounded-sm" style={{ background: CONDITION_COLOR[k] }} />{CONDITION_LABEL[k]}</td>
              {metricKeys.map((m) => {
                const r = view.conds[k].metrics.find((x) => x.metric === m && x.role === "evaluation");
                return <td key={m} className="py-4 text-right font-mono">{r && r.status === "completed" ? fmtNum(r.value) : <span className="text-base text-slate-500">{r ? r.status.replace("_", " ") : view.conds[k].status === "not_executed" ? "not executed" : "not run"}</span>}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {view.refinement && (
        <p className="mt-8 text-lg text-slate-400">C = best of {view.refinement.history.filter((h) => h.iteration > 0 && h.status === "scored").length} refinement iteration(s) (iteration {view.refinement.best_iteration}), selected by <span className="font-mono">{view.refinement.optimization_metric}</span> — which is excluded from the table above.</p>
      )}
    </div>
  );
  const integrity = (
    <div className="max-w-5xl">
      <Title kicker="Integrity">How to read these results</Title>
      <ul className="space-y-5 text-2xl leading-relaxed text-slate-300">
        <li>Every image and number comes from a recorded run with its seed, model version and code fingerprint.</li>
        <li>Stages that did not run are labelled — nothing is simulated or filled in.</li>
        <li>Refinement optimises one metric; we judge it with <i>different</i> metrics.</li>
        <li>One image is an illustration, not evidence — see Research Findings for n.</li>
      </ul>
    </div>
  );
  return [intro, pipeline, reference, compare, prompts, metrics, integrity];
}
