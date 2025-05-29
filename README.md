# Grouped Latent Attention


This repository contains the implementation of the modeling files for GLA-2 and GTA-G for the "[Hardware-Efficient Attention for Fast Decoding](https://arxiv.org/abs/2505.21487v1)". Refer to the [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main) repo in the `/hopper` section for the GLA kernels on H100. The optimized 256 head dimension is currently activated in FA3 for inference.<br>


**Grouped Latent Attention (GLA)** is a parallel-friendly form of latent attention that groups the latent into $$2d$$ head dimension, double of GQA half of MLA, and pairs this design with low-level optimizations, asynchronous execution and distributed offset calculation. GLA serves as a practical latent variant to efficiently scale across multiple GPUs by mitigating or eliminating latent KV cache duplication, and it effectively utilizes modern hardware (Hopper) with high arithmetic intensity during decoding. The following figure illustrates the data loading schematic of MLA (left) and GLA (right) for attention during decoding, highlighting reduced data movement and thereby boosting arithmetic intensity. GLA shards the latent data instead of duplicating it on every device, essentially mitigating the memory-bound bottleneck of sequential decoding. As a result, each device loads a smaller KV cache, which boosts throughput and lowers end-to-end latency.

<br>
<br>



![MLA](assets/mla_gla_schematic.png)


<br>
<br>



Roofline analysis of BF16 decoding on a single H100 80 GB SXM5. In this figure, the numeric suffix (e.g., GQ-*128*) denotes the number of query heads $h_q$. Left, $L_q{=}1$: With $h_q{=}128$, MLA reaches an arithmetic intensity of $\sim2h_q{=}256$, close to the H100 compute roof of $\sim 295$ FLOPs per byte, whereas GLA 128 with two latent heads stays on the memory-bandwidth roof (I/O roof) at $\sim h_q{=}128$, similar to MQA. Right, $L_q{=}2$: When the query length is two (as in speculative decoding), MLA with $h_q{=}128$ climbs beyond the roof and becomes compute bound, while GLA with two latent heads sits at the knee and can run up to two times faster.

![MLA](assets/roofline.png)



<br>
<br>



**Grouped-Tied Attention (GTA)**, which ties the key and value representations into a single shared state, leverages grouping to share the tied KV heads among a small set of query heads combined with partial RoPE; it roughly reduces KV cache size and improves the arithmetic intensity by up to a factor of two relative to its GQA counterpart with the same number of groups, while preserving quality and parallelism benefits. The following is an overview of GTA, where a single projection produces a *tied KV*. The full *tied KV* dimension is for the value. For the keys, half of the dimension is from *tied KV* (yellow) designated for no positional encoding, and the other half (red) comes from a separate single-head projection where RoPE is applied; this separate half is broadcast to all heads in the group and concatenated with the *tied KV* half. 
<br>
<br>

![GTA](assets/gta.png)









## Usage

GLA with FA3. Here we provide the pseudo code for running GLA during decoding with FA3 kernels. 

```python
from hopper.flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata

nheads_q  = 128
nheads_C  = 2  # latent (c_kv) heads
rope_dim  = 64
C_dim  = 256   # latent dimension

page_size = 64           


query_rope  = torch.randn(batch_size, seqlen_q, nheads_q, rope,   dtype=dtype, device=device)
query_nope  = torch.randn(batch_size, seqlen_q, nheads_q, C_dim, dtype=dtype, device=device)

key_rope = torch.randn(batch_size, seqlen, nheads_C, rope_dim,   dtype=dtype, device=device)
kv_C_cache = torch.randn(batch_size, seqlen, nheads_C, C_dim, dtype=dtype, device=device)

k_rope, kv_C_cache = [
    rearrange(t, "b (n p) h d -> (b n) p h d", p=page_size)
    for t in (k_rope, kv_C_cache)
]
page_table = rearrange(
    torch.arange(batch_size * seqlen // page_size, dtype=torch.int32, device=device),
    "(b s) -> b s", s=seqlen // page_size
)

cache_seqlens = torch.full((batch_size,), seqlen, dtype=torch.int32, device=device)

sched_meta = get_scheduler_metadata(
    batch_size, seqlen_q, seqlen, nheads_q, nheads_C, rope_dim,
    cache_seqlens, q.dtype, headdim_v=C_dim, page_size=page_size, causal=True
)

out = flash_attn_with_kvcache(
    query_rope, key_rope, kv_C_cache,
    cache_seqlens=cache_seqlens,
    num_splits=0,
    qv=query_nope,
    page_table=page_table,
    causal=True,
    scheduler_metadata=sched_meta
)
```

## Requirements

- Hopper GPUs
- CUDA 12.3 and above
    - We **highly** recommend **CUDA 12.8** for best performance.

## Acknowledgements
- [OLMo](https://github.com/allenai/OLMo) for the trainer codebase.


##  Citation
Bibtex:

```tex

@misc{zadouri2025hardwareefficientattentionfastdecoding,
      title={Hardware-Efficient Attention for Fast Decoding}, 
      author={Ted Zadouri and Hubert Strauss and Tri Dao},
      year={2025},
      eprint={2505.21487},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.21487}, 
}
