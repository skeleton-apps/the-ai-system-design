# What is KV Cache
During autoregressive decoding a decoder-only Transformer must compute, for every layer and every step $t$ 

```math
\begin{aligned}
Q_t &= x_t W_Q, \qquad
K_t &= x_t W_K, \qquad
V_t &= x_t W_V \\
Attention(Q_t, K_{1:t}, V_{1:t}) = softmax(\frac{Q_tK^T_{1:t}}{\sqrt{d}})V_{1:t}
\end{aligned}

Note: Only $Q_t$ involves the new token. The keys and values 






