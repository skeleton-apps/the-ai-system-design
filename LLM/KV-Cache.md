# What is KV Cache
During autoregressive decoding a decoder-only Transformer must compute, for every layer and every step $t$
$$Q_t = x_tW_Q, K_t = x_tW_K, V_t = x_tW_V$$

