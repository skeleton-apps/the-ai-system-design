# What is KV Caching
During autoregressive decoding a decoder-only Transformer must compute, for every layer and every step $t$ 

```math
\begin{aligned}
Q_t = x_t W_Q, \\
K_t = x_t W_K, \\
V_t = x_t W_V, \\
\textit{Attention}(Q_t, K_{1:t}, V_{1:t}) &= \mathrm{softmax}\left(\frac{Q_t K^T_{1:t}}{\sqrt{d}}\right) V_{1:t}
\end{aligned}
```
At step $t$, $Q_t$ only involves the new token $q_t$. The keys and values $K_{1:t-1}$, $V_{1:t-1}$ were already computed at earlier steps. KV-caching stores them once and simply appends $K_t$ and $V_t$

*A deep dive into autoregressive, token-by-token decoding*

> Autoregressive, token-by-token decoding is a process used in AI models, particularly large language models (LLMs) or other sequential models, to
> generate output sequences (like text or video) one item (or "token") at a time. 
```math
\begin{aligned}
P\!\bigl(x_{1:n}\bigr) \;=\; \prod_{t=1}^{n} P\!\bigl(x_t \,\big|\, x_{1:t-1}\bigr)
\end{aligned}
```
### Why do we cache **K** & **V**

| Tensor | Needed again? | Re-compute cost | Caching benefit |
|--------|---------------|-----------------|-----------------|
| **Q** | **No.** A query dies after its own step | `O(d)` (tiny) | None |
| **K** | **Yes.** Each new token must dot-product with every past key | `O(t d)` (grows) | Huge |
| **V** | **Yes.** Required to build the weighted sum once attention weights are known | `O(t d)` | Huge |
| **QKᵀ** | **No.** Counts will change the moment a new query arrives | `O(t d)` | None |

**Key takeaway:**  
> Past queries are “one-shot”, however past keys & values are needed for the current sequence. This Caching only K and V maximizes reuse.

---

## 1 What *exactly* is happening in autoregressive decoding?

At inference time a language model writes one token, feeds it back, then writes the next:

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


## Examples

## 1. Full Re-computation **(no KV-cache)**

All four queries $Q$, keys $K$, and values $V$ are re-evaluated.

```math
Q =
\begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1\\
2 & 2
\end{bmatrix},\;
K =
\begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1\\
2 & 2
\end{bmatrix},\;
V =
\begin{bmatrix}
0 & 1\\
1 & 0\\
1 & 1\\
2 & 2
\end{bmatrix}
```
### 1.1 Attention scores  

```math
QK^{\!\top}=
\begin{bmatrix}
1 & 0 & 1 & 2\\
0 & 1 & 1 & 2\\
1 & 1 & 2 & 4\\
2 & 2 & 4 & 8
\end{bmatrix}
```

### 1.2 Multiply by **V**  
Row 4 (Token 4) as an example:

```math
(QK^{\!\top})V \;=\;
\begin{bmatrix}
1 & 0 & 1 & 2\\
0 & 1 & 1 & 2\\
1 & 1 & 2 & 4\\
2 & 2 & 4 & 8
\end{bmatrix} . 
\begin{bmatrix}
0 & 1\\
1 & 0\\
1 & 1\\
2 & 2
\end{bmatrix} \;=\;
\boxed{
\begin{bmatrix}
5 & 6\\
6 & 5\\
11 & 11\\
22 & 22
\end{bmatrix}}
```


## 2&nbsp;▪&nbsp;Incremental decode **with KV-cache**

Only the latest token (Token 4) is computed; previous Key-Value pairs are **re-used**.

### 2.1 Cached tensors (after Tokens 1–3)

```math
K_\text{cache}=
\boxed{
\begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix}}
,\qquad
V_\text{cache}=
\boxed{
\begin{bmatrix}
0 & 1\\
1 & 0\\
1 & 1
\end{bmatrix}}
```

### 2.2 Newly-computed tensors (Token 4)

```math
Q_4=[2,\,2],\;
K_4=[2,\,2],\;
V_4=[2,\,2]
```


### 2.3 Concatenate cached + New

```math
K'=
\begin{bmatrix}
\color{purple}{1} & \color{purple}{0}\\
\color{purple}{0} & \color{purple}{1}\\
\color{purple}{1} & \color{purple}{1}\\
\color{orange}{2} & \color{orange}{2}
\end{bmatrix},
\quad
V'=
\begin{bmatrix}
\color{purple}{0} & \color{purple}{1}\\
\color{purple}{1} & \color{purple}{0}\\
\color{purple}{1} & \color{purple}{1}\\
\color{orange}{2} & \color{orange}{2}
\end{bmatrix}
```



### 2.4 Single-row attention

```math
Q_4 K'^{\!\top}=
[2,\,2,\,4,\,8]
```
```math
\begin{bmatrix}2 & 2 & 4 & 8\end{bmatrix}
\! \times V'
=2[0,1]+2[1,0]+4[1,1]+8[2,2]
=[22,\,22]
```
> **Legend**  
> *🟨 newly-computed at the current step* *🟪 fetched from KV-cache*  

---

### 🚀 Complexity Comparison

| Method | Per-step cost |
|--------|---------------|
| No cache | $O(n^2)$ |
| With cache | $O(n)$ |

KV-caching therefore keeps latency linear in sequence length during autoregressive decoding.
