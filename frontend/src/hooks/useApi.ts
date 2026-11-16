import { useCallback, useEffect, useRef, useState } from "react";

/** Fetch data, optionally re-polling while `shouldPoll(data)` is true. */
export function useApi<T>(fn: (() => Promise<T>) | null, deps: unknown[], opts?: { pollMs?: number; shouldPoll?: (d: T) => boolean }) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(!!fn);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const reload = useCallback(async () => {
    if (!fnRef.current) return;
    try {
      const d = await fnRef.current();
      setData(d);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setData(null);
    setLoading(!!fn);
    void reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  const pollMs = opts?.pollMs;
  const shouldPoll = opts?.shouldPoll;
  useEffect(() => {
    if (!pollMs || data === null) return;
    if (shouldPoll && !shouldPoll(data)) return;
    const t = setTimeout(() => void reload(), pollMs);
    return () => clearTimeout(t);
  }, [data, pollMs, shouldPoll, reload]);

  return { data, error, loading, reload, setData };
}

export const PENDING = new Set(["pending", "queued", "running", "generating", "scoring", "approved"]);
