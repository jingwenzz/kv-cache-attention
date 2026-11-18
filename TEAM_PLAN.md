# KV Cache Project Plan

The topic is the KV cache problem in on-device inference: during generation the cache grows linearly with seq len, and both memory capacity and bandwidth run out. Everything in this project follows from one formula:

cache bytes = 2 (K and V) x layers x kv_heads x head_dim x seq_len x bytes_per_element

Weights are constant, the cache is not. MHA, GQA, and MQA only differ in how many kv_heads they use, and FP8 just cuts bytes_per_element from 2 to 1.

## What we need to do

1. Read the five core papers and build a taxonomy: MQA, GQA, PagedAttention, StreamingLLM, H2O. For each one, a short note in our own words on what it compresses (number of heads, precision, number of tokens, or memory layout), when it takes effect, and where the quality cost shows up.
2. Run the three experiments (details in the code section below).
3. Make the slides. Mostly just show the experiment results: one curve plot, one heatmap, one takeaway sentence.
4. Write the paper, structure below. Experiments come first because the slides and the paper both need the results, so run experiments early and draft the paper skeleton while they run.

The critical analysis angle is settled, and it is the most valuable part of the paper: effective context window. Eviction methods (StreamingLLM, H2O) drop tokens permanently, so a needle that lands in a dropped region is gone for good. Quantization keeps every token and degrades gradually. The experiments exist to test this contrast.

## Taxonomy: representative papers (added, for the literature review)

The assignment asks for three categories: Cache Compression, Sparse Access, Architectural Simplification. Each one attacks a different factor of the formula above. The five core papers are in **bold**; the others are optional extra citations for the survey part, so we do not need to read them in full.

| Category | Which factor it reduces | Representative papers |
|---|---|---|
| Architectural Simplification (decided at design/training time) | kv_heads (and layers) | **MQA**: Shazeer 2019, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150). **GQA**: Ainslie et al. 2023, [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). Optional: MLA / DeepSeek-V2 2024, [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) (low-rank KV); Cross-Layer Attention, Brandon et al. 2024, [arXiv 2405.12981](https://arxiv.org/abs/2405.12981); sliding window in Mistral 7B, [arXiv 2310.06825](https://arxiv.org/abs/2310.06825) |
| Cache Compression (store less of what is cached) | bytes_per_element (quantization) and number of tokens (eviction) | Quantization: FP8 KV cache (vLLM / NVIDIA, no paper, see their docs); optional KIVI, Liu et al. 2024, [arXiv 2402.02750](https://arxiv.org/abs/2402.02750); KVQuant, Hooper et al. 2024, [arXiv 2401.18079](https://arxiv.org/abs/2401.18079). Eviction: **StreamingLLM**, Xiao et al. 2024, [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453); **H2O**, Zhang et al. 2023, [Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048); optional Scissorhands, [arXiv 2305.17118](https://arxiv.org/abs/2305.17118); SnapKV, [arXiv 2404.14469](https://arxiv.org/abs/2404.14469) |
| Sparse Access (keep the whole cache, read only part of it per step) | bytes read per step (bandwidth), not capacity | Optional: Quest, Tang et al. 2024, [arXiv 2406.10774](https://arxiv.org/abs/2406.10774) (query-aware page selection); SparQ Attention, Ribar et al. 2023, [arXiv 2312.04985](https://arxiv.org/abs/2312.04985); DuoAttention, Xiao et al. 2024, [arXiv 2410.10819](https://arxiv.org/abs/2410.10819) (retrieval heads vs streaming heads); Longformer, [arXiv 2004.05150](https://arxiv.org/abs/2004.05150) (static sparse patterns) |
| Orthogonal system layer (memory management, does not change the logical cache size) | fragmentation and over-reservation | **PagedAttention / vLLM**: Kwon et al. 2023, [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180). Optional: RadixAttention / SGLang (prefix sharing), [arXiv 2312.07104](https://arxiv.org/abs/2312.07104); FlexGen (offloading), [arXiv 2303.06865](https://arxiv.org/abs/2303.06865) |

For the effective context window analysis and the needle test:
- Liu et al. 2023, [Lost in the Middle](https://arxiv.org/abs/2307.03172): retrieval accuracy depends on where the information sits in the context.
- Hsieh et al. 2024, [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654): defines effective context length as the longest context where the model still performs at a threshold. We can borrow the definition (for example "success rate within 5 points of the FP16 baseline").
- Needle in a Haystack test: [Kamradt's repo](https://github.com/gkamradt/LLMTest_NeedleInAHaystack), the original harness we are re-implementing in `src/needle.py`.

How we classify (added). The criterion is what happens to a cache entry:
- Deleted for good (eviction) or stored in fewer bits (quantization): cache compression. Both reduce what is stored, so they save capacity and bandwidth.
- Kept in full but not read at this step: sparse access. It saves bandwidth only, not capacity, and a wrong selection can be corrected at the next step because the cache is intact.
- Cache size unchanged, waste around it removed (fragmentation, over-reservation): the orthogonal system layer. The assignment itself labels PagedAttention "memory management", which is why it is not in the compression family.
- Fewer KV heads by design: architectural simplification.

Boundary cases. StreamingLLM and H2O also behave like sparse attention (each step attends to a subset), but we classify by memory effect: their evicted entries are gone, so they belong with compression. DuoAttention straddles two families (streaming heads keep only a sink plus a window, retrieval heads keep everything).

Consequence for reading. With this grouping none of the five core papers falls in sparse access (MQA, GQA: architectural; StreamingLLM, H2O: compression; PagedAttention: system layer). We should add one representative for that family; Quest is the clearest (the cache stays intact, pages are selected per query).

### Cross-cutting comparison (added)

The table above groups papers by which factor they reduce. This second table compares the same methods along axes that cut across the families, and is the basis for the critical analysis section. The last column is what we expect, to be checked against our experiments, not a result.

| Method | When information is discarded | Recoverable? | Needs training or calibration | Saves | Expected place where quality drops |
|---|---|---|---|---|---|
| MQA / GQA | Before or during training (architecture) | No, it never existed | Yes: from scratch, or uptraining an MHA checkpoint | Capacity and bandwidth | Slight quality loss vs MHA; larger for MQA |
| StreamingLLM | At write, when a token leaves the window | No | No | Capacity and bandwidth | Needles outside the sink and the recent window |
| H2O | At write, by accumulated attention score | No | No | Capacity and bandwidth | A token judged unimportant early but needed by a later query |
| FP8 / KIVI-style quantization | At write | No, but every token is kept | No (FP8: a scale) | Capacity and bandwidth | Gradual: rounding noise accumulates over long contexts |
| Quest-style sparse access | At read | Yes, the cache is intact | No | Bandwidth only | A wrong page selection hurts that step only |
| PagedAttention | Nothing is discarded | Not applicable | No | Capacity (removes waste) | No quality cost; system complexity instead |

### Points we can argue (added; hypotheses, not results)

1. The methods act on different factors of the formula, so in principle they multiply (GQA x FP8 x paging). The interesting question is the interaction, not each method alone.
2. Possible interaction: in GQA each KV head is shared by several query heads, so quantization error in one KV head reaches several query heads. Whether GQA models are more sensitive to KV quantization than MHA models is open; we can only mention it as a question unless we have time to test it.
3. Ordering by recoverability: eviction should fail abruptly once the needle falls into dropped tokens, quantization should degrade gradually, and sparse access should sit in between. Our needle heatmaps (FP16, FP8, and the eviction baseline if we have time) test the first two, and this is the core of the effective context window argument.

## What the code needs

Repo: https://github.com/jingwenzz/kv-cache-attention

Stack is just Python + PyTorch + matplotlib. Small models are fine, we care about cache behavior, not model quality.

- `src/cache_size.py`: compute cache size from the formula, then check it against actual memory usage during inference. If the two disagree, be able to explain why (padding, allocator overhead, that kind of thing).
- `src/plot_curves.py`: plot cache GB vs seq len from 1K to 32K, MHA and GQA on the same figure, plus an FP16 vs FP8 comparison.
- `src/needle.py`: a needle in a haystack test harness. Insert the needle at different depths (0 to 100 percent) and different context lengths, measure retrieval success rate, output a heatmap.
- `src/fp8_cache.py`: in the generation loop, quantize and dequantize K/V in FP8 (e4m3), run the same needle test as the FP16 cache, and compare where the success rate starts to drop by length and depth.
- If we have spare time, add one eviction baseline (attention sink plus a recent window) to test the effective context window argument.
- `results/` holds the plots, `notes/` holds the per paper reading notes.

## What the paper needs

ACL template, double column, main text max 6 pages, references do not count. Five sections:

- Abstract: one sentence for the problem, one for the approach, one for the conclusion.
- Introduction: why on-device inference gets bottlenecked by the cache, bring in the formula.
- Taxonomy: the five papers in three families, architectural simplification (MQA, GQA), cache compression (quantization, PagedAttention), sparse access (StreamingLLM, H2O), with a comparison table.
- Critical analysis: the effective context window argument, why eviction and quantization degrade in fundamentally different ways.
- Experiments: the three experiments, figures first, text explains the figures. The curves show cache growth matches the formula, the heatmap shows FP8 degradation is gradual and where it starts.
- Conclusion: restate the main line in one takeaway.
- Contribution statement at the end saying who did what. This is explicitly required, do not forget it.

### Page budget and content plan (added, subject to change)

A rough allocation of the 6 main-text pages. Every number here can move once the figures exist.

| Section | Pages | What goes in |
|---|---|---|
| Abstract | 0.15 | Problem, approach, conclusion, one sentence each |
| Introduction | 0.6 | On-device motivation (cache outgrows the model), the formula, one concrete number (for example per-token cache of an MHA vs a GQA model), contributions in 2 to 3 bullets |
| Taxonomy | 1.5 | Taxonomy table (about half a page), then one short paragraph per family. Covers the five core papers; optional papers only in one-line mentions |
| Critical analysis | 1.0 | Effective context window: definition (RULER style), why eviction fails by dropping tokens permanently, why quantization degrades gradually, what the needle heatmap should look like under each, plus Lost in the Middle |
| Experiments | 2.2 | Setup (model, GPU, how the needle is built) about 0.4 page. Fig. 1: cache GB vs seq len, MHA vs GQA, with formula vs measured. Fig. 2: FP16 vs FP8 cache size. Fig. 3: needle heatmaps, FP16 vs FP8 (and eviction baseline if done). Text explains each figure; where the measurement disagrees with the formula, say why |
| Conclusion | 0.3 | One takeaway sentence plus limits (small model, one task type) |
| Contribution statement | 0.15 | Who did what |
| **Total** | **about 5.9** | |
| References | not counted | |

## Split (rough for now, easy to change)

- Taxonomy plus the taxonomy and analysis sections of the paper: one person
- Experiments 1 and 2 (cache size and curves): one person
- Experiment 3 (needle + FP8) plus the slides: one person

Whoever has a GPU starts the experiments first. We all read the full paper draft before submitting.
