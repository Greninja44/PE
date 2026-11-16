import { Card } from "../components/ui";
import { PageHeader } from "../layouts/AppLayout";

export function MethodologyPage() {
  return (
    <>
      <PageHeader title="Methodology" description="How the pipeline works, what each number means, and the controls that keep the results honest." />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Research question">
          <div className="prose-lab">
            <p>Given only a reference image, how faithfully can we reconstruct a text prompt that makes a diffusion model regenerate it? And does a structured, validated description outperform a plain caption — with or without iterative refinement?</p>
            <h3>Three conditions</h3>
            <ul>
              <li><b>A · Caption only</b> — the VLM's one-sentence caption used verbatim.</li>
              <li><b>D · Detailed prose prompt</b> — the same JSON attributes rendered as sentences (<span className="mono">natural_v1</span>).</li>
              <li><b>B · Structured prompt</b> — JSON attributes → fixed tag template → validated positive/negative prompt.</li>
              <li><b>C · Structured + refinement</b> — a refinement loop started from B. Reported as the <i>final</i> scored iteration; iteration 0 and the post-hoc <i>best</i> iteration are shown separately, the latter flagged as selection-biased.</li>
            </ul>
            <p>A, D and B are generated with the identical model, seed, steps, CFG and resolution; only the prompt differs. Paired comparisons only pair runs with identical settings on the same reference image.</p>
            <p>Human ratings (1–5, optionally blind) are stored in a separate table and never mixed into metric aggregates.</p>
          </div>
        </Card>
        <Card title="Pipeline">
          <div className="prose-lab">
            <ul>
              <li><b>Stage 1 — Analysis.</b> Qwen3-VL-2B (greedy decoding) returns subject, objects, spatial relations, composition, camera angle, lighting, colours, materials, textures, background and style as JSON. Raw output is stored; unparseable output is marked failed.</li>
              <li><b>Stage 2 — Prompting.</b> Versioned templates (structured_v1, subject_first_v1, natural_v1). Rule-based validation detects contradictions (e.g. daylight vs night), positive/negative overlap, duplicates, invalid relations and CLIP-window overflow. Only unambiguous fixes are automatic; the unfixed original is kept as the parent version.</li>
              <li><b>Stage 3 — Generation.</b> SD 1.5, SD-Turbo or SDXL via diffusers on the local inference worker (Apple MPS on this Mac, CUDA on an NVIDIA machine), with explicit seeds.</li>
              <li><b>Stage 4 — Evaluation.</b> Four independent metrics, each stored separately with its backend identifier.</li>
              <li><b>Stage 5 — Refinement.</b> Deterministic rules propose small edits; a human approves or rejects; hill-climbing from the best prompt with patience-based early stopping.</li>
            </ul>
          </div>
        </Card>
        <Card title="Metrics">
          <div className="prose-lab">
            <ul>
              <li><b>CLIP (ViT-L/14)</b> — cosine similarity of image embeddings; semantic content.</li>
              <li><b>DINOv2 (base)</b> — cosine similarity of self-supervised CLS features; object identity and structure.</li>
              <li><b>SSIM</b> — Gaussian-window structural similarity on luminance after centre-crop alignment. Very strict for prompt-based reconstruction (any layout shift lowers it). Verified to match scikit-image to machine precision.</li>
              <li><b>LPIPS (AlexNet)</b> — learned perceptual distance; lower is closer.</li>
            </ul>
          </div>
        </Card>
        <Card title="Optimisation ≠ evaluation">
          <div className="prose-lab">
            <p>Refinement optimises a single objective (default <span className="mono">CLIP ViT-B/32</span>). Reporting the same number as evidence of improvement would be circular (Goodhart's law), so independent evaluation uses a <i>different</i> set: CLIP ViT-L/14, DINOv2, SSIM and LPIPS. Each metric row is tagged with its role and the dashboard shows the two groups separately.</p>
          </div>
        </Card>
        <Card title="Reproducibility">
          <div className="prose-lab">
            <ul>
              <li>Every record stores the reference image SHA-256, prompt version lineage, model repository id, seed, steps, CFG, resolution, negative prompt, runtime (torch/diffusers versions, device — MPS/CUDA/CPU — dtype, generator device) and timing, plus measured peak worker memory.</li>
              <li>Seeds on Apple MPS and CPU use a CPU random generator, so the initial noise for a seed is identical on any machine; MPS kernels can still differ slightly from CUDA, so images are reproducible per device, not bit-identical across devices.</li>
              <li>A code fingerprint (git commit, dirty flag, SHA-256 of backend sources, package versions) is captured with every experiment and export.</li>
              <li>Reports are written with exclusive-create semantics: previous reports are never overwritten.</li>
              <li>Hosted inference providers are flagged “not reproducible” because seed handling cannot be verified.</li>
            </ul>
          </div>
        </Card>
        <Card title="Integrity rules & limitations">
          <div className="prose-lab">
            <ul>
              <li>No simulated outputs: if a provider is not configured the request is refused; if a metric cannot run its value is empty with a status.</li>
              <li>Automated tests use synthetic gradient images and hand-picked numbers that exercise control flow only; they never reach the database used for research.</li>
              <li>The token counter is an approximation of CLIP BPE; exact counts need the tokenizer on the worker.</li>
              <li>The refinement proposer is heuristic and depends on the VLM's descriptions of generated images, which may themselves be wrong.</li>
              <li>Small samples: report n and ranges, not just means.</li>
              <li>An earlier 12-image pilot (reported by the researcher; data not in this repository) found structured prompts did not consistently beat captions, and refinement gains may have been inflated by selection bias. This application is designed to test those findings again, not to overturn them by presentation.</li>
            </ul>
          </div>
        </Card>
      </div>
    </>
  );
}
