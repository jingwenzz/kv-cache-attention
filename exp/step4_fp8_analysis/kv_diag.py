"""Step 4a and 4b of the KV-cache FP8 study, as one script (CUDA, Apple MPS or CPU).

4a: teacher-forced log-prob and margin of the right answer under FP16 / FP8 e4m3 / FP8 e5m2 caches,
    on the same RULER prompts as the accuracy runs (same builder and seeds), paired against FP16.
4b: K/V numeric diagnostics: per-layer max |K|, |V|, share clamped by e4m3 (> 448) or in the
    subnormal range, measured FP8 relative error, K outlier-channel ratio.

The code is taken from the ruler_niah_*.ipynb notebooks; only device handling and config changed.
FP8 is simulated (fake quantization at cache write); on MPS and CPU the float8 cast is emulated
in float32, which the notebooks check matches the native cast bit for bit.

Examples
  python kv_diag.py --model Qwen/Qwen2.5-0.5B-Instruct --part 4b
  python kv_diag.py --model Qwen/Qwen2.5-0.5B-Instruct --part 4a --lengths 2048 4096 8192 --variants 5
  python kv_diag.py --model Qwen/Qwen3-4B --part 4a --tasks multikey single --lengths 4096 16384 --variants 5
  python kv_diag.py --model Qwen/Qwen2.5-0.5B-Instruct --part analyze     # tables from existing files only
Requires: pip install torch transformers accelerate pandas scipy tqdm wonderwords  (and git)
"""
import argparse, json, os, random, re, time, math, subprocess, glob
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--part", choices=["4a", "4b", "all", "analyze"], default="all")
p.add_argument("--tasks", nargs="+", default=["single", "multikey"])
p.add_argument("--modes", nargs="+", default=["fp16", "fp8_e4m3", "fp8_e5m2"])
p.add_argument("--lengths", nargs="+", type=int, default=[2048, 4096, 8192, 16384, 32768])
p.add_argument("--depths", nargs="+", type=int, default=[0, 25, 50, 75, 100])
p.add_argument("--variants", type=int, default=10, help="needles per (length, depth) cell, from variant 0")
p.add_argument("--stat-length", type=int, default=4096, help="4b prompt length")
p.add_argument("--stat-prompts", type=int, default=5, help="4b number of prompts")
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
args = p.parse_args()

# ---------------- device ----------------
if args.device == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEVICE = args.device
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
OOM_ERROR = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
def empty_cache():
    if DEVICE == "cuda": torch.cuda.empty_cache()
    elif DEVICE == "mps": torch.mps.empty_cache()
def sync():
    if DEVICE == "cuda": torch.cuda.synchronize()
    elif DEVICE == "mps": torch.mps.synchronize()
def reset_peak():
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()
def peak_mem():
    return torch.cuda.max_memory_allocated() if DEVICE == "cuda" else 0

RESULTS_DIR = args.out
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL = args.model
SHORT = MODEL.split("/")[-1]
TASKS = args.tasks
MODES = args.modes
N_DISTRACTORS = 3
TASK = TASKS[0]
CFG = dict(model_id=MODEL, depths=args.depths, seed=0, tag="fp16", max_new_tokens=1)
TF = dict(lengths=args.lengths, depths=args.depths, n_variants=args.variants)
PREFILL_EXACT = True
print(f"{SHORT} on {DEVICE} ({DTYPE}) | part {args.part} | tasks {TASKS} | lengths {TF['lengths']} | variants {TF['n_variants']} | out {RESULTS_DIR}")

def save_to_drive():   # the notebooks flush Google Drive here; nothing to do locally
    pass

def tf_path(mode): return f"{RESULTS_DIR}/ruler_{TASK}_{SHORT}_{mode}_tf.jsonl"

# ---------------- analysis (4a), runs on existing files ----------------
def analyze_4a():
    from scipy.stats import wilcoxon

    def wp(x):
        x = np.asarray(x); x = x[x != 0]
        return wilcoxon(x).pvalue if len(x) >= 10 else float("nan")

    for TASK_ in TASKS:
        global TASK
        TASK = TASK_
        print(f"\n========== {TASK} ==========")
        T = {m: pd.read_json(tf_path(m), lines=True, dtype=False).query("status == 'ok'") for m in MODES if os.path.exists(tf_path(m))}
        key = ["length", "depth", "variant"]
        f16 = T["fp16"].set_index(key)
        rows, rep = [], []
        for m in [x for x in T if x != "fp16"]:
            q = T[m].set_index(key).join(f16, rsuffix="_f", how="inner")
            q["d_lp"] = q.lp_sum - q.lp_sum_f
            q["d_min_margin"] = q.min_margin - q.min_margin_f
            cols = ["d_lp", "d_min_margin"]
            if "ret_margin" in q:
                q["d_ret"] = q.ret_margin - q.ret_margin_f; cols.append("d_ret")
            for L, g in list(q.groupby(level="length")) + [("all", q)]:
                row = dict(mode=m, length=L, n=len(g))
                for c in cols:
                    row[f"{c} median"] = round(g[c].median(), 3)
                    row[f"{c} worse"] = f"{(g[c] < 0).mean():.0%}"
                if L == "all":
                    for c in cols: row[f"{c} p"] = f"{wp(g[c]):.1e}"
                rows.append(row)
            if "ret_margin" in q:
                print(f"{m}: prompts preferring a distractor (ret_margin < 0): fp16 {(q.ret_margin_f < 0).mean():.1%}, {m} {(q.ret_margin < 0).mean():.1%}")
            for (val, mq, mf) in zip(q.value, q.margin, q.margin_f):
                for i in range(len(mq)):
                    after = i >= 2 and val[i - 1] == val[i - 2]
                    rep.append(dict(mode=m, after_repeat=after, d_margin=mq[i] - mf[i]))
        print(pd.DataFrame(rows).fillna("").to_string(index=False))
        R = pd.DataFrame(rep)
        print("\nmargin change by digit position type (mean, n)")
        print(R.groupby(["mode", "after_repeat"]).d_margin.agg(["mean", "median", "size"]).round(3).to_string())

if args.part == "analyze":
    analyze_4a()
    raise SystemExit

# ---------------- model ----------------
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import transformers
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
CACHE_KW = "cache" if int(transformers.__version__.split(".")[0]) >= 5 else "past_key_values"
print("transformers", transformers.__version__, "| layers", model.config.num_hidden_layers, "| kv heads", model.config.num_key_value_heads)

# ---------------- haystack (Paul Graham essays) ----------------
import glob, subprocess
NIAH_DIR = os.path.expanduser("~/.cache/kv_diag/niah")
ESSAY_DIR = f"{NIAH_DIR}/needlehaystack/PaulGrahamEssays"
if not os.path.isdir(ESSAY_DIR):
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack", NIAH_DIR], check=True)
files = sorted(glob.glob(f"{ESSAY_DIR}/*.txt"))
raw = " ".join(open(f, encoding="utf-8").read() for f in files)
raw = re.sub(r"\s+", " ", raw).strip()
SENTS = [s for s in re.split(r"(?<=[.!?])\s+", raw) if s]
SENT_TOK = [len(x) for x in tok(SENTS, add_special_tokens=False)["input_ids"]]
print(len(files), "essays,", len(SENTS), "sentences,", sum(SENT_TOK), "tokens")
assert sum(SENT_TOK) > 2 * max(TF["lengths"]), "haystack too short"

# ---------------- RULER prompt builder (same as the notebooks) ----------------
from wonderwords import random_word
_adjs = random_word._get_words_from_text_file("adjectivelist.txt")
_nouns = random_word._get_words_from_text_file("nounlist.txt")
WORDS = sorted(set(f"{a}-{n}" for a in _adjs for n in _nouns))

TEMPLATE = ("A special magic number is hidden within the following text. Make sure to memorize it. "
            "I will quiz you about the number afterwards.\n{context}\n"
            "What is the special magic number for {query} mentioned in the provided text?")
ANSWER_PREFIX = "The special magic number for {query} mentioned in the provided text are"
NEEDLE = "One of the special magic numbers for {key} is: {value}."

def ntok(s):
    return len(tok(s, add_special_tokens=False)["input_ids"])

def build_prompt(target_len, depth, variant, task=None, seed=CFG["seed"]):
    task = task or TASK
    n_needles = 1 + (N_DISTRACTORS if task == "multikey" else 0)
    r = random.Random(f"needles-{seed}-{variant}")
    keys = r.sample(WORDS, n_needles)
    values = []
    while len(values) < n_needles:
        x = str(r.randint(1000000, 9999999))
        if x not in values: values.append(x)
    needles = [NEEDLE.format(key=k, value=v) for k, v in zip(keys, values)]
    query, value = keys[0], values[0]

    overhead = ntok(TEMPLATE.format(context="", query=query)) + ntok(ANSWER_PREFIX.format(query=query)) + 30
    budget = target_len - overhead - sum(ntok(n) for n in needles)
    rng = random.Random(f"hay-{seed}-{target_len}-{variant}")
    start = rng.randrange(len(SENTS))
    idx, total = [], 0
    while total < budget:
        i = (start + len(idx)) % len(SENTS)
        idx.append(i); total += SENT_TOK[i]
    chosen = [SENTS[i] for i in idx]

    k = round(depth / 100 * len(chosen))
    actual_depth = 100 * sum(SENT_TOK[i] for i in idx[:k]) / total
    pr = random.Random(f"pos-{seed}-{target_len}-{variant}")
    inserts = [(k, 0, needles[0])] + [(pr.randrange(len(chosen) + 1), 1, n) for n in needles[1:]]
    out, ins = [], sorted(inserts)
    j = 0
    for pos in range(len(chosen) + 1):
        while j < len(ins) and ins[j][0] == pos:
            out.append(ins[j][2]); j += 1
        if pos < len(chosen): out.append(chosen[pos])
    context = " ".join(out)
    prompt = TEMPLATE.format(context=context, query=query)
    return prompt, dict(value=value, key=query, prefix=ANSWER_PREFIX.format(query=query),
                        distractors=values[1:], actual_depth=actual_depth)

# ---------------- generation ----------------
import transformers
CACHE_KW = "cache" if int(transformers.__version__.split(".")[0]) >= 5 else "past_key_values"

@torch.no_grad()
def answer(prompt, prefix, max_new_tokens, cache=None):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefix
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    empty_cache(); reset_peak(); sync()
    t0 = time.perf_counter()
    kw = {CACHE_KW: cache} if cache is not None else {}
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None, **kw)
    sync()
    dt = time.perf_counter() - t0
    peak = peak_mem()
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True), ids.shape[1], dt, peak

# ---------------- FP8 / INT quantizers ----------------
FP8_FMT = {"fp8_e4m3": dict(dtype="float8_e4m3fn", mb=3, emin=-6, vmax=448.0),
           "fp8_e5m2": dict(dtype="float8_e5m2",   mb=2, emin=-14, vmax=57344.0)}

def fp8_native(x, fmt):
    c = FP8_FMT[fmt]
    return x.float().clamp(-c["vmax"], c["vmax"]).to(getattr(torch, c["dtype"])).to(x.dtype)

def fp8_emulated(x, fmt):
    c = FP8_FMT[fmt]
    xf = x.float().clamp(-c["vmax"], c["vmax"])
    e = torch.floor(torch.log2(xf.abs().clamp(min=2.0 ** (c["emin"] - c["mb"]))))
    e = e.clamp(min=c["emin"])
    step = 2.0 ** (e - c["mb"])
    return (torch.round(xf / step) * step).clamp(-c["vmax"], c["vmax"]).to(x.dtype)

def quant_int(x, bits):
    xf = x.float()
    mn, mx = xf.amin(-1, keepdim=True), xf.amax(-1, keepdim=True)
    qmax = 2 ** bits - 1
    scale = ((mx - mn) / qmax).clamp(min=1e-8)
    return (torch.round((xf - mn) / scale).clamp(0, qmax) * scale + mn).to(x.dtype)

# choose native or emulated FP8 for this GPU
probe = torch.cat([torch.randn(1 << 15), torch.randn(1 << 15) * 1e-3, torch.randn(1 << 15) * 100]).half().to(DEVICE)
IMPL = {}
for fmt in FP8_FMT:
    emu = fp8_emulated(probe, fmt)
    try:
        nat = fp8_native(probe, fmt)
        IMPL[fmt] = fp8_native
        print(f"{fmt}: native cast works; differs from emulation in {int((nat != emu).sum())} of {probe.numel()} values")
    except Exception as e:
        IMPL[fmt] = fp8_emulated
        print(f"{fmt}: native cast failed ({type(e).__name__}); using emulation")

def get_qfn(mode):
    if mode in FP8_FMT:
        f = IMPL[mode]; return lambda x: f(x, mode)
    if mode == "int8": return lambda x: quant_int(x, 8)
    if mode == "int4": return lambda x: quant_int(x, 4)
    raise ValueError(mode)

# ---------------- quantizing cache ----------------
from transformers import DynamicCache

class QuantCache(DynamicCache):
    def __init__(self, qfn, prefill_exact=True, record=False):
        try:
            super().__init__(config=model.config)
        except TypeError:
            super().__init__()
        self.qfn, self.prefill_exact = qfn, prefill_exact
        self.stats = [] if record else None

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        first = self.get_seq_length(layer_idx) == 0
        qk, qv = self.qfn(key_states), self.qfn(value_states)
        if self.stats is not None:
            kf, vf = key_states.float(), value_states.float()
            self.stats.append((((qk.float() - kf).norm() / kf.norm()).item(),
                               ((qv.float() - vf).norm() / vf.norm()).item()))
        k, v = super().update(qk, qv, layer_idx, *args, **kwargs)
        if first and self.prefill_exact:
            return key_states, value_states
        return k, v

def qcache(mode, record=False):
    return lambda: QuantCache(get_qfn(mode), PREFILL_EXACT, record)

# ---------------- 4a: teacher-forced scoring ----------------
def tf_path(mode): return f"{RESULTS_DIR}/ruler_{TASK}_{SHORT}_{mode}_tf.jsonl"

def new_cache(mode):
    if mode == "fp16":
        try: return DynamicCache(config=model.config)
        except TypeError: return DynamicCache()
    return qcache(mode)()

_FWD_KW = []
def fwd(ids, cache, keep):
    # forward with a given cache; checks the cache was actually used (an unknown kwarg would be ignored silently)
    names = _FWD_KW or [CACHE_KW, "past_key_values" if CACHE_KW == "cache" else "cache"]
    for name in names:
        before = cache.get_seq_length()
        try:
            out = model(input_ids=ids, use_cache=True, logits_to_keep=keep, **{name: cache})
        except TypeError:
            continue
        if cache.get_seq_length() == before + ids.shape[1]:
            if not _FWD_KW: _FWD_KW.append(name)
            return out.logits
    raise RuntimeError("could not pass the cache to model.forward")

@torch.no_grad()
def score_trial(L, d, v, mode):
    prompt, meta = build_prompt(L, d, v)
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + meta["prefix"]
    pre = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    n_pre = pre.shape[1]
    cache = new_cache(mode)
    last = fwd(pre, cache, 1)[0, -1].float()
    cands = [meta["value"]] + meta["distractors"]
    per = []
    for ci, val in enumerate(cands):
        full = tok(text + " " + val, add_special_tokens=False)["input_ids"]
        assert full[:n_pre] == pre[0].tolist(), "prefix tokenization changed when the answer was appended"
        ans = full[n_pre:]
        if ci > 0:
            cache.crop(-(cache.get_seq_length() - n_pre))      # negative = drop that many tokens (works in 4.x and 5.x)
            assert cache.get_seq_length() == n_pre
        lg = fwd(torch.tensor([ans], device=DEVICE), cache, len(ans))[0].float()
        preds = torch.cat([last[None], lg[:-1]], 0)               # row j predicts ans[j]
        tgt = torch.tensor(ans, device=DEVICE)
        lp = torch.log_softmax(preds, -1).gather(1, tgt[:, None])[:, 0]
        right = preds.gather(1, tgt[:, None])[:, 0]
        other = preds.scatter(1, tgt[:, None], float("-inf")).max(-1).values
        dig = [i for i, t in enumerate(ans) if tok.decode([t]).strip().isdigit()]
        per.append(dict(lp=[round(lp[i].item(), 4) for i in dig], margin=[round((right[i] - other[i]).item(), 3) for i in dig]))
    del cache; empty_cache()
    rec = dict(task=TASK, mode=mode, length=L, depth=d, variant=v, value=meta["value"],
               lp=per[0]["lp"], margin=per[0]["margin"],
               lp_sum=round(sum(per[0]["lp"]), 4), min_margin=min(per[0]["margin"]), status="ok")
    if len(per) > 1:
        rec["distractor_lp_sum"] = [round(sum(p["lp"]), 4) for p in per[1:]]
        rec["ret_margin"] = round(rec["lp_sum"] - max(rec["distractor_lp_sum"]), 4)
    return rec

def tf_grid(mode):
    path = tf_path(mode); done = set()
    if os.path.exists(path):
        done = {(r["length"], r["depth"], r["variant"]) for r in map(json.loads, open(path))}
    todo = [(L, d, v) for L in TF["lengths"] for d in TF["depths"] for v in range(TF["n_variants"]) if (L, d, v) not in done]
    oom_at = []
    with open(path, "a") as f:
        for L, d, v in tqdm(todo, desc=f"{TASK} {mode}"):
            if any(L >= x for x in oom_at):
                rec = dict(task=TASK, mode=mode, length=L, depth=d, variant=v, status="oom")
            else:
                try:
                    rec = score_trial(L, d, v, mode)
                except OOM_ERROR:
                    empty_cache(); oom_at.append(L)
                    rec = dict(task=TASK, mode=mode, length=L, depth=d, variant=v, status="oom")
            f.write(json.dumps(rec) + "\n"); f.flush()
    return pd.read_json(path, lines=True, dtype=False)

if args.part in ("4a", "all"):
    TASK = TASKS[0]
    r = score_trial(min(TF["lengths"]), 50, 0, "fp16")
    print("check | forward cache kwarg:", _FWD_KW, "| digits", r["value"], "| margin", r["margin"], "| ret_margin", r.get("ret_margin"))
    for TASK in TASKS:
        for mode in MODES:
            d = tf_grid(mode)
            ok = d[d.status == "ok"]
            print(f"{TASK} {mode}: {len(ok)} scored, median lp_sum {ok.lp_sum.median():.3f}, median min margin {ok.min_margin.median():.2f}")
    analyze_4a()

# ---------------- 4b: K/V numeric diagnostics ----------------
E4M3, E5M2 = FP8_FMT["fp8_e4m3"], FP8_FMT["fp8_e5m2"]

class StatCache(DynamicCache):
    def __init__(self):
        try: super().__init__(config=model.config)
        except TypeError: super().__init__()
        self.rows = []
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self.get_seq_length(layer_idx) == 0:                     # prompt K/V only
            row = dict(layer=layer_idx)
            for name, x in [("K", key_states), ("V", value_states)]:
                a = x.float().abs()
                nz = a[a > 0]
                row[f"{name} max"] = a.max().item()
                row[f"{name} >448 (e4m3 clamp)"] = (a > E4M3["vmax"]).float().mean().item()
                row[f"{name} <2^-6 (e4m3 subnormal)"] = (nz < 2.0 ** E4M3["emin"]).float().mean().item()
                row[f"{name} ->0 e4m3"] = (nz < 2.0 ** (E4M3["emin"] - E4M3["mb"] - 1)).float().mean().item()
                row[f"{name} <2^-14 (e5m2 subnormal)"] = (nz < 2.0 ** E5M2["emin"]).float().mean().item()
                for mode in ["fp8_e4m3", "fp8_e5m2"]:
                    qx = get_qfn(mode)(x).float()
                    row[f"{name} relerr {mode[4:]}"] = ((qx - x.float()).norm() / x.float().norm()).item()
                if name == "K":
                    ch = a.amax(dim=(0, 2)).flatten()                # per (head, dim) channel
                    row["K outlier ratio"] = (ch.max() / ch.median()).item()
            self.rows.append(row)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)

if args.part in ("4b", "all"):
    TASK = "single"
    stat_rows = []
    for v in range(args.stat_prompts):
        prompt, meta = build_prompt(args.stat_length, 50, v)
        c = StatCache()
        answer(prompt, meta["prefix"], 1, c)
        stat_rows += [dict(r, variant=v) for r in c.rows]
    S = pd.DataFrame(stat_rows)
    per_layer = S.groupby("layer").agg({c: ("max" if c.endswith("max") or "ratio" in c else "mean") for c in S.columns if c not in ("layer", "variant")})
    pd.set_option("display.width", 250)
    print(per_layer.round(5).to_string())
    summary = {c: (per_layer[c].max() if c.endswith("max") or "ratio" in c else per_layer[c].mean()) for c in per_layer.columns}
    print("\nall layers:", {k: round(v, 6) for k, v in summary.items()})
    per_layer.to_csv(f"{RESULTS_DIR}/kv_stats_{SHORT}.csv")
    print("saved", f"{RESULTS_DIR}/kv_stats_{SHORT}.csv")
