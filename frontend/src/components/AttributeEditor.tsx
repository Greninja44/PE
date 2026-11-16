import { Plus, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { FIELD_LABEL } from "../lib/format";
import type { Attributes } from "../types/api";
import { Field } from "./ui";

const TEXT_FIELDS = ["subject", "art_style", "composition", "camera_angle", "lighting", "background"] as const;
const LIST_FIELDS = ["secondary_objects", "colors", "materials", "textures", "extra_positive", "extra_negative"] as const;

function ListInput({ value, onChange, placeholder }: { value: string[]; onChange: (v: string[]) => void; placeholder?: string }) {
  const [text, setText] = useState(value.join(", "));
  useEffect(() => {
    const parsed = text.split(",").map((s) => s.trim()).filter(Boolean);
    if (parsed.join("|") !== value.join("|")) setText(value.join(", "));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  return (
    <input className="input" value={text} placeholder={placeholder}
      onChange={(e) => { setText(e.target.value); onChange(e.target.value.split(",").map((s) => s.trim()).filter(Boolean)); }} />
  );
}

export function AttributeEditor({ value, onChange, disabled }: { value: Attributes; onChange: (a: Attributes) => void; disabled?: boolean }) {
  const set = <K extends keyof Attributes>(k: K, v: Attributes[K]) => onChange({ ...value, [k]: v });
  return (
    <fieldset disabled={disabled} className="space-y-4">
      <div className="grid gap-3 md:grid-cols-2">
        {TEXT_FIELDS.map((f) => (
          <Field key={f} label={FIELD_LABEL[f]} className={f === "subject" ? "md:col-span-2" : ""}>
            <input className="input" value={value[f]} onChange={(e) => set(f, e.target.value)} />
          </Field>
        ))}
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {LIST_FIELDS.map((f) => (
          <Field key={f} label={FIELD_LABEL[f]} hint="comma-separated">
            <ListInput value={value[f]} onChange={(v) => set(f, v)} />
          </Field>
        ))}
      </div>
      <div>
        <div className="mb-1.5 flex items-center justify-between">
          <span className="label">{FIELD_LABEL.spatial_relationships}</span>
          <button type="button" className="btn-ghost px-2 py-1 text-xs"
            onClick={() => set("spatial_relationships", [...value.spatial_relationships, { subject: "", relation: "next to", object: "" }])}>
            <Plus className="h-3.5 w-3.5" />Add
          </button>
        </div>
        <div className="space-y-2">
          {value.spatial_relationships.map((r, i) => (
            <div key={i} className="grid grid-cols-[1fr_0.8fr_1fr_auto] gap-2">
              {(["subject", "relation", "object"] as const).map((k) => (
                <input key={k} className="input" placeholder={k} value={r[k]} onChange={(e) => {
                  const next = value.spatial_relationships.map((x, j) => (j === i ? { ...x, [k]: e.target.value } : x));
                  set("spatial_relationships", next);
                }} />
              ))}
              <button type="button" className="btn-ghost px-2" title="Remove"
                onClick={() => set("spatial_relationships", value.spatial_relationships.filter((_, j) => j !== i))}>
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          ))}
          {!value.spatial_relationships.length && <p className="text-sm text-slate-500">No relationships.</p>}
        </div>
      </div>
    </fieldset>
  );
}

export function AttributeView({ a }: { a: Attributes }) {
  const rows: [string, string][] = [];
  for (const f of ["subject", ...LIST_FIELDS.slice(0, 1), "art_style", "composition", "camera_angle", "lighting", "background"] as const) {
    const v = a[f as keyof Attributes];
    rows.push([FIELD_LABEL[f], Array.isArray(v) ? (v as string[]).join(", ") : String(v ?? "")]);
  }
  rows.push([FIELD_LABEL.colors, a.colors.join(", ")], [FIELD_LABEL.materials, a.materials.join(", ")], [FIELD_LABEL.textures, a.textures.join(", ")]);
  return (
    <div className="space-y-3">
      <dl className="grid gap-x-6 gap-y-2.5 sm:grid-cols-2">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt className="label text-[10px]">{k}</dt>
            <dd className="mt-0.5 text-sm text-slate-200">{v || <span className="text-slate-600">—</span>}</dd>
          </div>
        ))}
      </dl>
      {a.spatial_relationships.length > 0 && (
        <div>
          <p className="label mb-1 text-[10px]">{FIELD_LABEL.spatial_relationships}</p>
          <ul className="flex flex-wrap gap-1.5">
            {a.spatial_relationships.map((r, i) => (
              <li key={i} className="rounded-md border border-ink-600 bg-ink-800 px-2 py-1 text-xs text-slate-300">
                {r.subject} <span className="text-accent-300">{r.relation}</span> {r.object}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
