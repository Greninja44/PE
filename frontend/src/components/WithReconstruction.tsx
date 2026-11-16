import { ImagePlus } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { PENDING, useApi } from "../hooks/useApi";
import { useActiveReconstruction } from "../hooks/useActiveReconstruction";
import { api } from "../services/api";
import type { Reconstruction } from "../types/api";
import { Empty, ErrorText, Spinner } from "./ui";

/** True if anything attached to the reconstruction is still in flight (drives polling). */
export function hasPending(r: Reconstruction): boolean {
  return r.analyses.some((a) => PENDING.has(a.status))
    || r.generations.some((g) => PENDING.has(g.status) || g.metrics.some((m) => PENDING.has(m.status)))
    || r.refinement_sessions.some((s) => s.iterations.some((i) => PENDING.has(i.status)));
}

export function WithReconstruction({ children }: { children: (r: Reconstruction, reload: () => Promise<void>) => ReactNode }) {
  const { activeId } = useActiveReconstruction();
  const q = useApi(activeId ? () => api.reconstruction(activeId) : null, [activeId], { pollMs: 2500, shouldPoll: hasPending });
  if (!activeId) {
    return (
      <Empty icon={<ImagePlus className="h-8 w-8" />} title="No reconstruction selected">
        Pick one from the selector above, or <Link to="/new" className="text-accent-300 underline">upload a reference image</Link>.
      </Empty>
    );
  }
  if (q.error) return <ErrorText error={q.error} />;
  if (!q.data) return <div className="flex justify-center py-16"><Spinner className="h-6 w-6" /></div>;
  return <>{children(q.data, q.reload)}</>;
}
