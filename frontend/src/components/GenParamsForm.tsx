import { Dice5 } from "lucide-react";
import type { GenParams, ModelSpec } from "../types/api";
import { Field } from "./ui";

export const defaultParams = (m?: ModelSpec): GenParams => ({
  model_key: m?.key ?? "sd_turbo", seed: 42, steps: m?.default_steps ?? 1, cfg_scale: m?.default_cfg ?? 0,
  width: m?.default_size ?? 512, height: m?.default_size ?? 512, negative_prompt: "",
});

export function GenParamsForm({ models, value, onChange, showNegative = true }: {
  models: ModelSpec[]; value: GenParams; onChange: (p: GenParams) => void; showNegative?: boolean;
}) {
  const spec = models.find((m) => m.key === value.model_key);
  const set = <K extends keyof GenParams>(k: K, v: GenParams[K]) => onChange({ ...value, [k]: v });
  const num = (s: string) => (s === "" ? null : Number(s));
  return (
    <div className="space-y-3">
      <Field label="Model" hint={spec ? `${spec.hf_id} · ${spec.vram_note}` : undefined}>
        <select className="input" value={value.model_key} onChange={(e) => onChange({ ...defaultParams(models.find((m) => m.key === e.target.value)), seed: value.seed, negative_prompt: value.negative_prompt })}>
          {models.map((m) => <option key={m.key} value={m.key}>{m.label}{m.downloaded === false ? " — not downloaded" : ""}{m.recommended_16gb === false ? " — not recommended on 16 GB" : ""}</option>)}
        </select>
      </Field>
      {spec?.notes && <p className="text-xs text-slate-400">{spec.notes}</p>}
      {spec?.downloaded === false && <p className="text-xs text-amber-300">Weights are not in the local cache: the first run downloads them (~{spec.est_gb ?? "?"} GB) and will take a while. Prefer <span className="mono">scripts/download_models.py</span> beforehand.</p>}
      {spec?.recommended_16gb === false && <p className="text-xs text-amber-300">This model needs ≈{spec.est_gb} GB for weights alone; on a 16 GB Mac it is likely to swap or fail. It has not been verified on this machine.</p>}
      <div className="grid grid-cols-2 gap-3">
        <Field label="Seed">
          <div className="flex gap-1.5">
            <input className="input font-mono" type="number" min={0} value={value.seed} onChange={(e) => set("seed", Number(e.target.value))} />
            <button type="button" className="btn-secondary px-2.5" title="Random seed" onClick={() => set("seed", Math.floor(Math.random() * 2 ** 31))}><Dice5 className="h-4 w-4" /></button>
          </div>
        </Field>
        <Field label={`Steps${spec ? ` (${spec.steps_range[0]}–${spec.steps_range[1]})` : ""}`}>
          <input className="input font-mono" type="number" value={value.steps ?? ""} min={spec?.steps_range[0]} max={spec?.steps_range[1]} onChange={(e) => set("steps", num(e.target.value))} />
        </Field>
        <Field label={`CFG scale${spec ? ` (${spec.cfg_range[0]}–${spec.cfg_range[1]})` : ""}`}>
          <input className="input font-mono" type="number" step={0.5} value={value.cfg_scale ?? ""} min={spec?.cfg_range[0]} max={spec?.cfg_range[1]} disabled={spec?.cfg_range[1] === 0} onChange={(e) => set("cfg_scale", num(e.target.value))} />
        </Field>
        <Field label="Resolution">
          <div className="flex items-center gap-1.5">
            <input className="input font-mono" type="number" step={8} value={value.width ?? ""} onChange={(e) => set("width", num(e.target.value))} />
            <span className="text-slate-500">×</span>
            <input className="input font-mono" type="number" step={8} value={value.height ?? ""} onChange={(e) => set("height", num(e.target.value))} />
          </div>
        </Field>
      </div>
      {showNegative && (
        <Field label="Negative prompt override" hint="Empty = use the prompt version's own negative prompt.">
          <textarea className="input min-h-16 font-mono text-[13px]" value={value.negative_prompt} onChange={(e) => set("negative_prompt", e.target.value)} />
        </Field>
      )}
    </div>
  );
}
