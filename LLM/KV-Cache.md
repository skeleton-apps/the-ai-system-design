# What is KV Cache
During autoregressive decoding a decoder-only Transformer must compute, for every layer and every step $t$
$$
\begin{aligned}
Q_t &= x_t W_Q, \qquad
K_t &= x_t W_K, \qquad
V_t &= x_t W_V
\end{aligned}
$$

