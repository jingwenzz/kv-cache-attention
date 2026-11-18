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

## Split (rough for now, easy to change)

- Taxonomy plus the taxonomy and analysis sections of the paper: one person
- Experiments 1 and 2 (cache size and curves): one person
- Experiment 3 (needle + FP8) plus the slides: one person

Whoever has a GPU starts the experiments first. We all read the full paper draft before submitting.
