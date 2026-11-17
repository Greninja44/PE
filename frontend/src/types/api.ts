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
  is_manual: boolean; runtime_info?: Record<string, unknown> | null; created_at: string;
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
  height: number | null; negative_prompt: string; lora?: string | null;
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
  notes: string; vram_note: string; est_gb?: number; recommended_16gb?: boolean; downloaded?: boolean | null;
}

export interface MetricInfo {
  key: string; label: string; higher_is_better: boolean; execution: string; backend: string; range: string;
  description: string; available_in_api_process: boolean; api_process_reason: string; default_role: string;
}

export interface WorkerMemory {
  process_gb?: number; process_metric?: string; system_total_gb?: number; system_available_gb?: number;
  system_used_percent?: number; swap_used_gb?: number; mps_allocated_gb?: number; mps_driver_gb?: number;
  cuda_allocated_gb?: number;
}

export interface WorkerCapabilities {
  job_types?: string[];
  runtime?: { torch?: string | null; selected_device?: string; gpu?: string; mps_available?: boolean;
    cuda_available?: boolean; mps_recommended_max_gb?: number; mps_fallback_enabled?: boolean; vram_gb?: number };
  memory?: WorkerMemory;
  resident_models?: { key: string; category: string; est_gb: number | null; load_s: number; idle_s: number;
    process_gb_after_load: number | null }[];
  model_events?: { t: number; event: string; key: string; reason?: string; process_gb?: number }[];
  packages?: Record<string, boolean>;
}

export interface WorkerInfo {
  id: string; name: string; online: boolean; last_seen_at: string | null; capabilities: WorkerCapabilities;
}

export interface SystemInfo {
  app: string; vlm_providers: Provider[]; local_vlm_models: Record<string, string>; generation_providers: Provider[];
  local_vlm_cached?: Record<string, boolean | null>;
  lora_adapters?: { id: string; base_model: string | null; steps?: number; rank?: number; evaluated: boolean }[];
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
  conditions: { A_caption: Generation | null; D_detailed?: Generation | null; B_structured: Generation | null; C_refined: Generation | null };
  refinement: { session_id: string; shown_iteration?: number; shown?: string; best_iteration: number | null; optimization_metric: string;
    history: { iteration: number; status: string; score: number | null }[] } | null;
}

export interface DemoBundleSummary { id: string; name: string; created_at: string; conditions: Record<string, string> }

export interface DemoCondition {
  status: string; prompt?: string; negative_prompt?: string; prompt_version?: number; prompt_source?: string;
  template_id?: string; provider?: string; model_key?: string; model_id?: string; params?: GenParams;
  duration_s?: number | null; runtime_info?: Record<string, unknown> | null;
  metrics?: { metric: string; role: string; status: string; value: number | null; higher_is_better: boolean; backend: string }[];
  image?: string;
  refinement?: { shown_iteration?: number; shown?: string; best_iteration: number | null; optimization_metric: string; history: { iteration: number; status: string; score: number | null }[] };
}

export interface DemoBundle {
  schema: string; name: string; reference_image: string; created_at: string; provenance: string;
  code_fingerprint: Record<string, unknown>; conditions: Partial<Record<"A_caption" | "D_detailed" | "B_structured" | "C_refined", DemoCondition>>;
}

export interface Summary { n: number; mean: number; sd: number | null; min: number; max: number }

export interface Aggregate {
  groups: Record<string, { selection_biased: boolean; metrics: Record<string, Summary>; human_similarity: Summary | null;
    other: Record<string, Summary> }>;
  paired: { a: string; b: string; metric: string; higher_is_better: boolean; n_pairs: number; b_better: number;
    b_worse: number; ties: number; mean_diff_b_minus_a: number; selection_biased: boolean }[];
  metrics: Record<string, { label: string; higher_is_better: boolean }>;
  condition_order: string[]; n_reconstructions: number; n_completed_generations: number; notes: string[];
}

export interface ComparisonRow {
  generation_id: string; reconstruction_id: string; condition: string | null; groups: string[]; status: Status;
  output_image_url: string | null;
  created_at: string; prompt_version: number; prompt_source: string; template_id: string; prompt: string;
  prompt_chars: number; prompt_words: number; prompt_clip_tokens_approx: number; model_key: string;
  params: Record<string, number | null>; config_key: string; device: string | null; generation_s: number | null;
  peak_memory_gb: number | null; refinement_session_id: string | null; refinement_iteration: number | null;
  refinement_iterations_executed: number | null;
  metrics: Record<string, { value: number | null; status: Status; higher_is_better: boolean }>;
  optimization: { metric: string; value: number | null; status: Status }[];
  human: { n: number; mean_similarity: number | null };
}

export interface ComparisonTable {
  reconstruction_id: string; rows: ComparisonRow[]; config_groups: Record<string, string[]>;
  metrics: Record<string, { label: string; higher_is_better: boolean }>; condition_order: string[]; selection_biased: string[];
}

export interface HumanRating {
  id: string; generation_id: string; rater: string; similarity: number; subject_match: number | null;
  style_match: number | null; blind: boolean; notes: string; kind: "human"; created_at: string;
}

export interface Stats {
  comparison: Aggregate;
  by_metric_and_condition: Record<string, Record<string, { n: number; mean: number; min: number; max: number }>>;
  counts: { reconstructions: number; completed_metrics: number };
}

// ------------------------------------------------------------------------------------------- fine-tuning
export interface FinetuneStatus {
  state: "preparing" | "training" | "completed" | "stopped" | "failed"; step: number; total_steps: number;
  message?: string; loss_ema?: number; val_loss?: number; val_loss_initial?: number; sec_per_step?: number;
  eta_s?: number; epochs?: number; duration_s?: number; finished_at?: string;
  memory?: { peak_process_gb?: number; system_available_gb?: number };
  memory_peak?: { peak_process_gb?: number; peak_mps_driver_gb?: number };
}

export interface FinetunePaired {
  n: number; lora_better: number; lora_worse: number; ties: number; mean_base: number | null; mean_lora: number | null;
  mean_diff_lora_minus_base: number | null; sd_diff: number | null; higher_is_better: boolean;
}

export interface FinetuneRun {
  id: string; base_model: string | null; adapter_available: boolean; status: FinetuneStatus;
  config: { rank?: number; alpha?: number; lr?: number; steps?: number; grad_accum?: number; seed?: number; device?: string;
    frozen_dtype?: string; n_train?: number; n_val_used?: number; created_at?: string; method?: string;
    dataset_card?: FinetuneDatasetCard; code_fingerprint?: { git_commit?: string | null } };
  evaluation: { n: number; settings: Record<string, unknown>; paired: Record<string, FinetunePaired>; created_at: string; notes: string[] } | null;
}

export interface FinetuneRunDetail extends FinetuneRun {
  train_log: { step: number; loss: number; loss_ema: number; lr: number; seconds: number }[];
  val_log: { step: number; val_loss: number }[];
  checkpoints: string[];
}

export interface FinetuneEval {
  run_id: string; n: number; settings: Record<string, unknown>; paired: Record<string, FinetunePaired>;
  nsfw_flagged: { base: number; lora: number }; notes: string[]; created_at: string;
  records: { index: number; prompt: string; seed: number; ref_file: string;
    base: Record<string, number | string | boolean | null> & { file: string }; lora: Record<string, number | string | boolean | null> & { file: string } }[];
}

export interface FinetuneDatasetCard {
  source: string; note: string; built_at: string; parts: number[]; filters: Record<string, unknown>; candidates: number;
  dropped: Record<string, number>; counts: Record<string, number>; prompt_groups: Record<string, number>;
  prompt_group_overlap: Record<string, number>; split_method: string;
}
