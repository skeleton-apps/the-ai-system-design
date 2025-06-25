# Flash Attention 
The self-attention mechanism is the cornerstone of transformer models, enabling them to capture long-range dependencies in sequences. However, as the sequence length $(L)$ increases, the computational and memory requirements of standard attention grow quadratically $O(L^2)$. This quadratic scaling presents a significant bottleneck, particularly for large language models (LLMs) that process very long sequences.

## The Attention Equation
The scaled dot-product attention equation creates large intermeiate score matrix $S=K^T$ which has $L\times L$ dimension.
```math
\begin{aligned}
Q = x W, \\
K = x W, \\
V = x W, \\
\textit{Attention}(Q, K, V) &= \mathrm{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V
\end{aligned}
```
GPU memory Levels
* High Bandwidth Memory (HBM): Large capacity, but relatively slow access
* SRAM (Shared Memory): Small capacity, but significantly faster access (on-chip).
The attention calculation often involves repeatedly reading and writing these large intermediate matrices (score matrix) between HBM and SRAM. This constant data transfer is the main cause of slow performance and out-of-memory errors for long sequences.
## How Flash Attention Attentions tackles this issue
Flash Attention tackles this memory bottleneck through two main techniques
### 1. Tiling (Block-wise Computation):
Instead of computing the entire attention matrix $(QK^T)$ at once, Flash Attention breaks down the Query $(Q)$, Key $(K)$, and Value $(V)$ matrices into smaller blocks (tiles). These blocks are loaded from the slower HBM into the faster SRAM. The attention calculation (matrix multiplication and softmax) is then performed on these smaller blocks entirely within SRAM. This minimizes the number of times data needs to be transferred between HBM and SRAM
### 2. Kernel Fusion and Online Softmax:
Flash Attention fuses multiple operations (like matrix multiplication, softmax, and dropout) into a single GPU kernel. This means that intermediate results don't need to be written back to HBM after each step, further reducing memory traffic.

A key challenge with tiling the softmax operation is that softmax requires summing across an entire row. Flash Attention uses an "online softmax" trick. It accumulates the sum of exponentials (the denominator of softmax) and the maximum value of the scores in a numerically stable way as it processes the tiles. This allows it to compute the exact softmax result without ever materializing the full attention score matrix in memory.

In the backward pass (for gradients), instead of storing the large attention matrix, Flash Attention recomputes the necessary blocks on the fly, saving even more memory.

Flash Attention directly targets this fundamental problem by redesigning the attention algorithm to be "memory-aware" and maximize the use of fast on-chip memory (SRAM) while minimizing data movement to/from slow off-chip memory (HBM).

* [Git](https://github.com/Dao-AILab/flash-attention)
* [Paper](https://arxiv.org/abs/2205.14135)


