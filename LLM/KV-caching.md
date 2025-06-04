# KV Cache with PagedAttention
High throughput LLM models require batching large ammount of requests at a time. However exisiting system struggled the Key-Value (KV cache) memory is very large. To address this problem this paper proposed PagedAttention mechanism. The proposed vLLM achieves near-zero waste in KV cache memory and enables flexable sharing of KV cache within and across requests which further reduces memory usage. Evaluations show 2-4x improvement of popular LLM's



