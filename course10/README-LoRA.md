I'll rewrite your adapter training example using LoRA (Low-Rank Adaptation) instead. Think of LoRA like adding lightweight "skill modules" to your existing model - instead of heavy adapters, you're adding nimble rank-decomposed matrices.Here's the key difference between your original adapter approach and this LoRA version:

**Original Adapters vs LoRA - The Analogy:**
- **Your adapters** were like adding complete "skill modules" with bottlenecks - full neural pathways that process information
- **LoRA** is like adding lightweight "adjustment dials" - just two small matrices (A and B) that create targeted modifications

**How LoRA Works:**
Instead of `W_new = W_original + adapter_output`, LoRA does:
`W_new = W_original + (alpha/rank) * A @ B`

Where A and B are much smaller matrices (rank 8 vs your bottleneck size 30).

**Key Benefits in this example:**
1. **Memory efficient**: Your bottleneck adapters might add ~30 parameters per dimension, while LoRA adds only `rank * (in_dim + out_dim)` parameters
2. **Mathematically elegant**: The low-rank decomposition captures the most important directions of change
3. **Easy to merge**: You can later merge LoRA weights directly into the original weights if needed

**Python-specific improvements I added:**
- Used `AdamW` with weight decay (better for fine-tuning)
- Added parameter counting and percentage calculation
- Included a `save_lora_weights()` function to save only the tiny LoRA parameters

The training loop works identically - only the LoRA matrices get updated while the original "foundation" stays frozen, like adjusting the tuning pegs on a guitar without changing the strings themselves.

