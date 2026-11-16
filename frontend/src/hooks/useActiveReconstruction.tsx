import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { useSearchParams } from "react-router-dom";

interface Ctx { activeId: string | null; setActiveId: (id: string | null) => void }
const ActiveCtx = createContext<Ctx>({ activeId: null, setActiveId: () => {} });
const KEY = "pe.activeReconstruction";

function readStored(): string | null {
  try { return localStorage.getItem(KEY); } catch { return null; }
}

export function ActiveReconstructionProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useSearchParams();
  const [activeId, setActive] = useState<string | null>(params.get("r") ?? readStored());

  useEffect(() => {
    const fromUrl = params.get("r");
    if (fromUrl && fromUrl !== activeId) setActive(fromUrl);
  }, [params, activeId]);

  const setActiveId = useCallback((id: string | null) => {
    setActive(id);
    try { if (id) localStorage.setItem(KEY, id); else localStorage.removeItem(KEY); } catch { /* storage blocked */ }
    const next = new URLSearchParams(params);
    if (id) next.set("r", id); else next.delete("r");
    setParams(next, { replace: true });
  }, [params, setParams]);

  return <ActiveCtx.Provider value={{ activeId, setActiveId }}>{children}</ActiveCtx.Provider>;
}

export const useActiveReconstruction = () => useContext(ActiveCtx);
