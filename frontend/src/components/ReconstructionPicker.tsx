import { Layers } from "lucide-react";
import { useApi } from "../hooks/useApi";
import { useActiveReconstruction } from "../hooks/useActiveReconstruction";
import { api } from "../services/api";

export function ReconstructionPicker() {
  const { activeId, setActiveId } = useActiveReconstruction();
  const list = useApi(() => api.reconstructions(), [activeId]);
  const active = list.data?.find((r) => r.id === activeId);
  return (
    <div className="flex min-w-0 items-center gap-3">
      {active ? <img src={active.image.url} alt="" className="h-8 w-8 rounded-md border border-ink-600 object-cover" />
        : <Layers className="h-5 w-5 text-slate-500" />}
      <select className="input max-w-xs py-1.5" value={activeId ?? ""} onChange={(e) => setActiveId(e.target.value || null)}>
        <option value="">Select a reconstruction…</option>
        {list.data?.map((r) => <option key={r.id} value={r.id}>{r.name} · {r.id.slice(0, 6)}</option>)}
      </select>
    </div>
  );
}
