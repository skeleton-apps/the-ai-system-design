# Grouped-Query Attention (GQA).

## Background 
To understand GQA, it's essential to first grasp the basic multi head attention mechanism. The Multi-Head Attention (MHA), introduced in the "Attention Is All You Need" paper, MHA enhances the attention mechanism by running multiple "attention heads" in parallel. Each head has its own independent set of learnable linear projection matrices $W^Q$, $W^K$, $W^V$ to transform the input $Q$, $K$ and $V$ vectors. This allows the model to capture diffnt types of relationships and attend to information from various "representation subspaces" simultaneously. The outputs from all heads are then concatenated and linearly transformed to produce the final output.

However the multi-head attention requires high memory bandwidht and computational cost, especially during autoregressive inference in LLMs. This is because each head maintains its own set of Key and Value caches (KV cache), which can grow very large for long sequences.


## Multi-Query Attention (MQA)
MQA was proposed to address the memory bottleneck of MHA. In MQA, all attention heads share the same single Key $K$ and value $V$ head, while still maintaining 
separate query $Q$ heads. This apporach significantly reduces memory bandwidth usage and speeds up inference because only one set of K and V vectors needs to be loaded and stored.

However it Can lead to a degradation in model quality compared to MHA, as sharing K and V across all query heads might limit the model's ability to capture diverse information.
