"""Hardware-aware inference runtime: device selection (Apple MPS → CPU, CUDA kept for NVIDIA workers),
dtype policy, cache release, memory diagnostics and sequential model lifecycle.

Imported only by the inference worker. The API process never imports torch."""
