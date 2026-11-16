import type {
  Analysis, Attributes, BuiltPrompt, Comparison, DemoBundle, DemoBundleSummary, Experiment, GenParams, Generation,
  Job, MetricResult, PromptDiff, PromptVersion, Reconstruction, ReconstructionSummary, RefinementSession, Stats,
  SystemInfo,
} from "../types/api";

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: init?.body instanceof FormData ? init?.headers : { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch { /* non-JSON error */ }
    throw new ApiError(res.status, msg);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  system: () => request<SystemInfo>("/api/system"),
  stats: () => request<Stats>("/api/stats"),
  jobs: (status?: string) => request<Job[]>(`/api/jobs${status ? `?status=${status}` : ""}`),
  cancelJob: (id: string) => post<Job>(`/api/jobs/${id}/cancel`),
  retryJob: (id: string) => post<Job>(`/api/jobs/${id}/retry`),

  reconstructions: () => request<ReconstructionSummary[]>("/api/reconstructions"),
  reconstruction: (id: string) => request<Reconstruction>(`/api/reconstructions/${id}`),
  createReconstruction: (file: File, name: string, notes: string, experimentId?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("name", name);
    fd.append("notes", notes);
    if (experimentId) fd.append("experiment_id", experimentId);
    return request<ReconstructionSummary>("/api/reconstructions", { method: "POST", body: fd });
  },

  requestAnalysis: (rid: string, body: { provider: string; model?: string | null; mode: string; image_id?: string }) =>
    post<Analysis>(`/api/reconstructions/${rid}/analyses`, body),
  manualAnalysis: (rid: string, body: { attributes?: Attributes; caption?: string; annotator: string }) =>
    post<Analysis>(`/api/reconstructions/${rid}/analyses/manual`, body),

  promptsFromAnalysis: (rid: string, analysis_id: string, template_id: string) =>
    post<PromptVersion[]>(`/api/reconstructions/${rid}/prompts/from-analysis`, { analysis_id, template_id }),
  createPromptVersion: (rid: string, body: {
    parent_id?: string | null; attributes: Attributes; template_id: string; note: string;
    positive_override?: string | null; negative_override?: string | null;
  }) => post<PromptVersion>(`/api/reconstructions/${rid}/prompts`, body),
  previewPrompt: (attributes: Attributes, template_id: string, caption?: string) =>
    post<{ original: BuiltPrompt; validated: BuiltPrompt }>("/api/prompts/preview", { attributes, template_id, caption }),
  promptDiff: (a: string, b: string) => request<PromptDiff>(`/api/prompts/diff?a=${a}&b=${b}`),

  generate: (rid: string, body: { prompt_version_id: string; provider: string; params: Partial<GenParams>; condition?: string | null }) =>
    post<Generation>(`/api/reconstructions/${rid}/generations`, body),
  evaluate: (gid: string, metrics: string[], role = "evaluation") =>
    post<MetricResult[]>(`/api/generations/${gid}/evaluate`, { metrics, role }),

  startComparison: (rid: string, body: { caption_prompt_version_id?: string; structured_prompt_version_id?: string; provider: string; params: Partial<GenParams> }) =>
    post<Generation[]>(`/api/reconstructions/${rid}/comparison`, body),
  comparison: (rid: string) => request<Comparison>(`/api/reconstructions/${rid}/comparison`),
  exportDemoBundle: (rid: string) => post<{ bundle: string }>(`/api/reconstructions/${rid}/demo-bundle`),
  demoBundles: () => request<DemoBundleSummary[]>("/api/demo/bundles"),
  demoBundle: (id: string) => request<DemoBundle>(`/api/demo/bundles/${id}`),
  demoFileUrl: (id: string, name: string) => `/api/demo/bundles/${id}/files/${name}`,

  startRefinement: (rid: string, body: Record<string, unknown>) =>
    post<RefinementSession>(`/api/reconstructions/${rid}/refinement`, body),
  refinement: (sid: string) => request<RefinementSession>(`/api/refinement/${sid}`),
  propose: (sid: string, max_edits: number) => post<RefinementSession>(`/api/refinement/${sid}/propose`, { max_edits }),
  approve: (iid: string, edit_indices: number[] | null) =>
    post<RefinementSession>(`/api/refinement/iterations/${iid}/approve`, { edit_indices }),
  reject: (iid: string, reason: string) => post<RefinementSession>(`/api/refinement/iterations/${iid}/reject`, { reason }),
  stopRefinement: (sid: string) => post<RefinementSession>(`/api/refinement/${sid}/stop`),

  experiments: () => request<Experiment[]>("/api/experiments"),
  createExperiment: (body: { name: string; description: string; hypothesis: string }) =>
    post<Experiment>("/api/experiments", body),
  writeReport: (eid: string) => post<Record<string, string>>(`/api/experiments/${eid}/reports`),
  reports: () => request<{ name: string; size_bytes: number }[]>("/api/reports"),
  fingerprint: () => request<Record<string, unknown>>("/api/fingerprint"),
};
