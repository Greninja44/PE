import { ImagePlus, Upload } from "lucide-react";
import { useState, type DragEvent } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { PageHeader } from "../layouts/AppLayout";
import { useApi } from "../hooks/useApi";
import { useActiveReconstruction } from "../hooks/useActiveReconstruction";
import { api } from "../services/api";
import { Card, ErrorText, Field } from "../components/ui";

export function NewReconstructionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [notes, setNotes] = useState("");
  const [experimentId, setExperimentId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const exps = useApi(() => api.experiments(), []);
  const { setActiveId } = useActiveReconstruction();
  const navigate = useNavigate();
  const { search } = useLocation();

  const pick = (f: File | undefined | null) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    if (!name) setName(f.name.replace(/\.[^.]+$/, ""));
  };
  const onDrop = (e: DragEvent) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); };

  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.createReconstruction(file, name, notes, experimentId || undefined);
      setActiveId(r.id);
      const next = new URLSearchParams(search);
      next.set("r", r.id);
      navigate({ pathname: "/analysis", search: next.toString() });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader title="New reconstruction" description="Upload the reference image whose prompt you want to reconstruct. Files are verified, content-addressed by SHA-256 and stored under experiments/originals/." />
      <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
        <Card title="Reference image">
          <label onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop}
            className={`flex aspect-[4/3] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition ${drag ? "border-accent-400 bg-accent-500/5" : "border-ink-600 hover:border-ink-500"}`}>
            {preview ? <img src={preview} alt="preview" className="max-h-full max-w-full rounded-lg object-contain p-2" /> : (
              <div className="text-center text-slate-400">
                <ImagePlus className="mx-auto mb-3 h-10 w-10 text-slate-500" />
                <p className="font-medium text-slate-200">Drop an image or click to browse</p>
                <p className="mt-1 text-sm">PNG, JPEG or WebP · up to 20 MB</p>
              </div>
            )}
            <input type="file" accept="image/png,image/jpeg,image/webp" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
          </label>
        </Card>
        <Card title="Details">
          <div className="space-y-4">
            <Field label="Name"><input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. red bicycle, brick wall" /></Field>
            <Field label="Notes"><textarea className="input min-h-24" value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Source, licence, why this image was chosen…" /></Field>
            <Field label="Experiment" hint="Optional — group this reconstruction into an experiment for export.">
              <select className="input" value={experimentId} onChange={(e) => setExperimentId(e.target.value)}>
                <option value="">None</option>
                {exps.data?.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
              </select>
            </Field>
            <ErrorText error={error} />
            <button className="btn-primary w-full py-2.5" disabled={!file || busy} onClick={submit}>
              <Upload className="h-4 w-4" />{busy ? "Uploading…" : "Create reconstruction"}
            </button>
          </div>
        </Card>
      </div>
    </>
  );
}
