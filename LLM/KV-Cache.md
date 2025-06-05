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

```mermaid
graph LR
A[Square Rect] -- Link text --> B((Circle))
A --> C(Round Rect)
B --> D{Rhombus}
C --> D
```
```mermaid
%%{init: {'theme':'default','logLevel':'fatal'}}%%
flowchart LR
    %% ---------- (a) ----------
    subgraph A["(a)"]
        direction TB

        %% input X
        subgraph aX["X"]
            direction LR
            ax1[" "]:::in
            ax2[" "]:::in
            ax3[" "]:::in
            ax4[" "]:::in
            ax5[" "]:::in
        end

        %% output Y (teacher-forcing, token-by-token)
        subgraph aY["Y"]
            direction LR
            bos(["[bos]"]):::special
            ay1(["Y₁"]):::out
            ay2(["Y₂"]):::out
            ay3(["Y₃"]):::out
            eos(["[eos]"]):::special

            bos --> ay1
            ay1 --> ay2
            ay2 --> ay3
            ay3 --> eos

            %% curved feedback arrows (dotted)
            ay1 -.-> ay2
            ay2 -.-> ay3
            ay3 -.-> eos
        end

        aX --> aY
    end

    %% ---------- (b) ----------
    subgraph B["(b)"]
        direction TB

        %% input X
        subgraph bX["X"]
            direction LR
            bx1[" "]:::in
            bx2[" "]:::in
            bx3[" "]:::in
            bx4[" "]:::in
            bx5[" "]:::in
        end

        %% output Y (known length)
        subgraph bY["Y"]
            direction LR
            by1(["Y₁"]):::out
            by2(["Y₂"]):::out
            by3(["Y₃"]):::out
            by4(["Y₄"]):::out

            by1 --> by2 --> by3 --> by4
        end

        bX -->|output length| bY
    end

    %% ---------- styles ----------
    classDef in     fill:#b7eec5,stroke:#333;
    classDef out    fill:#f8baba,stroke:#333;
    classDef special fill:#ffffff,stroke:#333,stroke-dasharray:5 3;
```



