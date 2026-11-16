export type Status = "pending" | "queued" | "running" | "completed" | "failed" | "unavailable";

export interface ImageInfo {
  id: string; kind: string; filename: string; width: number; height: number; mime: string;
  sha256: string; size_bytes: number; url: string; created_at: string;
}

export interface SpatialRelation { subject: string; relation: string; object: string }

export interface Attributes {
  subject: string;
  secondary_objects: string[];
  spatial_relationships: SpatialRelation[];
  composition: string;
  camera_angle: string;
  lighting: string;
  colors: string[];
  materials: string[];
  textures: string[];
  background: string;
  art_style: string;
  extra_positive: string[];
  extra_negative: string[];
}

export interface Analysis {
  id: string; reconstruction_id: string; image_id: string; mode: "structured" | "caption";
  provider: string; model: string; status: Status; result: Attributes | null; caption: string | null;
  raw_output: string | null; error: string | null; duration_s: number | null; job_id: string | null;
  is_manual: boolean; created_at: string;
}

export interface Issue {
  code: string; severity: "error" | "warning" | "info"; message: string; fields: string[]; terms: string[];
  auto_fixable: boolean;
}

export interface Validation {
  ok: boolean; issues: Issue[]; approx_clip_tokens: number; token_limit: number; fixes_applied: string[];
  segments?: { field: string; text: string }[];
}

export interface PromptVersion {
  id: string; reconstruction_id: string; version: number; parent_id: string | null; analysis_id: string | null;
  source: string; template_id: string; attributes: Attributes; positive: string; negative: string;
  validation: Validation | null; note: string; created_at: string;
}

export interface GenParams {
  model_key: string; seed: number; steps: number | null; cfg_scale: number | null; width: number | null;
  height: number | null; negative_prompt: string;
}

export interface MetricResult {
  id: string; metric: string; label: string; role: "optimization" | "evaluation"; status: Status;
  value: number | null; higher_is_better: boolean; backend: string; detail: string | null; job_id: string | null;
  created_at: string;
}

export interface Generation {
  id: string; reconstruction_id: string; prompt_version_id: string; prompt_version: number; positive: string;
  condition: string | null; provider: string; model_key: string; model_id: string; params: GenParams;
  status: Status; output_image: ImageInfo | null; error: string | null; duration_s: number | null;
  runtime_info: Record<string, unknown> | null; job_id: string | null; metrics: MetricResult[]; created_at: string;
}

export interface Edit { op: string; field: string; value: string; old_value?: string | null; reason: string }

export interface Iteration {
  id: string; iteration: number; status: string; prompt_version_id: string | null; generation_id: string | null;
  optimization_score: number | null; decided_at: string | null; created_at: string;
  proposal: {
    baseline?: boolean; edits: Edit[]; rationale: string; proposer?: string; template_id?: string;
    used_generated_analysis?: boolean; base_iteration?: number; rejection_reason?: string;
    proposed_edits?: Edit[]; approved_edit_indices?: number[];
  } | null;
}

export interface RefinementSession {
  id: string; reconstruction_id: string; base_prompt_version_id: string; optimization_metric: string;
  max_iterations: number; patience: number; min_delta: number; status: string; stop_reason: string | null;
  best_iteration_id: string | null; created_at: string; iterations: Iteration[];
  generation_config: { provider: string; params: GenParams; analysis_provider: string | null; warnings?: string[] };
}

export interface ReconstructionSummary {
  id: string; name: string; notes: string; experiment_id: string | null; image: ImageInfo; created_at: string;
  n_analyses: number; n_prompts: number; n_generations: number; n_completed_generations: number;
}

export interface Reconstruction extends ReconstructionSummary {
  analyses: Analysis[]; prompt_versions: PromptVersion[]; generations: Generation[];
  refinement_sessions: RefinementSession[];
}

export interface Provider {
  name: string; execution: "inline" | "worker"; description: string; model?: string; available: boolean; reason: string;
}

export interface ModelSpec {
  key: string; hf_id: string; label: string; family: string; default_steps: number; default_cfg: number;
  default_size: number; steps_range: [number, number]; cfg_range: [number, number]; max_side: number;
  notes: string; vram_note: string;
}

export interface MetricInfo {
  key: string; label: string; higher_is_better: boolean; execution: string; backend: string; range: string;
  description: string; available_in_api_process: boolean; api_process_reason: string; default_role: string;
}

export interface WorkerInfo {
  id: string; name: string; online: boolean; last_seen_at: string | null; capabilities: Record<string, unknown>;
}

export interface SystemInfo {
  app: string; vlm_providers: Provider[]; local_vlm_models: Record<string, string>; generation_providers: Provider[];
  models: ModelSpec[]; metrics: MetricInfo[]; templates: { id: string; description: string }[];
  jobs: Record<string, number>; workers: WorkerInfo[]; gpu_worker_online: boolean;
  worker_registration_enabled: boolean; code: { git_commit: string | null; git_dirty: boolean | null; source_sha256: string };
}

export interface Job {
  id: string; type: string; status: string; payload: Record<string, unknown>; error: string | null; attempts: number;
  max_attempts: number; worker_id: string | null; progress: number; progress_message: string; created_at: string;
  started_at: string | null; finished_at: string | null;
}

export interface Experiment {
  id: string; name: string; description: string; hypothesis: string; config: Record<string, unknown>;
  code_fingerprint: Record<string, unknown>; created_at: string; reconstructions: ReconstructionSummary[];
}

export interface BuiltPrompt {
  template_id: string; positive: string; negative: string; negative_terms: string[];
  segments: { field: string; text: string }[]; validation: Validation; attributes: Attributes;
}

export interface DiffOp { op: "equal" | "insert" | "delete"; text: string }

export interface PromptDiff {
  a: PromptVersion; b: PromptVersion; positive: DiffOp[]; negative: DiffOp[];
  attributes: { field: string; before: unknown; after: unknown }[];
}

export interface Comparison {
  reconstruction: ReconstructionSummary;
  conditions: { A_caption: Generation | null; B_structured: Generation | null; C_refined: Generation | null };
  refinement: { session_id: string; best_iteration: number; optimization_metric: string;
    history: { iteration: number; status: string; score: number | null }[] } | null;
}

export interface DemoBundleSummary { id: string; name: string; created_at: string; conditions: Record<string, string> }

export interface DemoCondition {
  status: string; prompt?: string; negative_prompt?: string; prompt_version?: number; prompt_source?: string;
  template_id?: string; provider?: string; model_key?: string; model_id?: string; params?: GenParams;
  duration_s?: number | null; runtime_info?: Record<string, unknown> | null;
  metrics?: { metric: string; role: string; status: string; value: number | null; higher_is_better: boolean; backend: string }[];
  image?: string;
  refinement?: { best_iteration: number; optimization_metric: string; history: { iteration: number; status: string; score: number | null }[] };
}

export interface DemoBundle {
  schema: string; name: string; reference_image: string; created_at: string; provenance: string;
  code_fingerprint: Record<string, unknown>; conditions: Record<"A_caption" | "B_structured" | "C_refined", DemoCondition>;
}

export interface Stats {
  by_metric_and_condition: Record<string, Record<string, { n: number; mean: number; min: number; max: number }>>;
  counts: { reconstructions: number; completed_metrics: number };
}
