# What is KV Caching
During autoregressive decoding a decoder-only Transformer must compute, for every layer and every step $t$ 

```math
\begin{aligned}
Q_t &= x_t W_Q, \\
K_t &= x_t W_K, \\
V_t &= x_t W_V, \\
\textit{Attention}(Q_t, K_{1:t}, V_{1:t}) &= \mathrm{softmax}\left(\frac{Q_t K^T_{1:t}}{\sqrt{d}}\right) V_{1:t}
\end{aligned}
```

**Note**: Only $Q_t$ involves the new token. The keys and values $K_{1:t-1}$, $V_{1:t-1}$ were already computed at earlier steps. KV-caching stores them once and simply appends $K_t$ and $V_t$

Without caching the per-token cost will be $O(t\times L \times d^2)$
With KV-cache it drops to $O(L \times d^2)$

## When and when not to use KV-Caching
When to use KV-Caching
1. Autoregressive token-by-token decoding
2. High perfmance inference
3. Long prompt and generated output use cases

When NOT to use KV-caching
1. Full Sequence processing in parallel (Non-autoregressive)
2. When the context changes frequently
3. Fine-Tuning or Traning 

## Tread-offs
KV-cache trades GPU memory for a drastic drop in per-token compute, enabling low-latency
for long-context LLM inference.





