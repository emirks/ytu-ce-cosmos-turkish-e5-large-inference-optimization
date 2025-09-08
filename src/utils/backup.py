# pip install -U transformers torch accelerate numpy tqdm pyarrow
# Optional speed-ups (script falls back if missing): pip install -U optimum

import os, time, json, math, torch, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from tqdm import tqdm
import pyarrow.parquet as pq

# =======================
# Config
# =======================
PARQUET_PATH = "chunks.parquet"
# If None, auto-detect among these candidates in order
TEXT_COL = "mdtext"  # e.g., "text_snippet" or "content" or "mdtext"
OUT_EMB_PATH = "embeddings.fp16.npy"
OUT_META_PATH = "meta.json"

MODEL_ID = "ytu-ce-cosmos/turkish-e5-large"
DEVICE = "cuda"
PREFIX = "passage: "  # e5 docs convention; change to "" if undesired
MAX_LEN = 512
BATCH = 2048  # starting guess; autotuner will adjust
BATCH_LIMIT = 16384
NUM_WORKERS = 12
PREFETCH = 8
AUTO_TUNE_BATCH = True
USE_COMPILE = (
    False  # <- keep False for stability; you can turn True AFTER autotune (see below)
)
DTYPE = torch.float16

# Backends / env
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass


# =======================
# Utilities
# =======================


def detect_text_column(pf: pq.ParquetFile) -> str:
    candidates = [
        "text",
        "text_snippet",
        "content",
        "mdtext",
        "chunk_text",
        "body",
    ]
    cols = set(pf.schema.names)
    for c in candidates:
        if c in cols:
            return c
    # fallback: choose first string-like column if known, else raise
    for name in pf.schema.names:
        # Can't inspect logical type easily here; try best effort
        return name
    raise RuntimeError("No columns found in parquet to use as text.")


def collate(batch, tokenizer):
    texts = [f"{PREFIX}{t}" for (t, _idx) in batch]
    indices = [int(_idx) for (_t, _idx) in batch]
    toks = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
    )
    return toks, indices


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-6)
    return summed / counts


def maybe_bettertransformer(model):
    try:
        from optimum.bettertransformer import BetterTransformer

        model = BetterTransformer.transform(model, keep_original_model=False)
        return model, True
    except Exception:
        return model, False


def maybe_compile(model):
    try:
        model = torch.compile(model, mode="max-autotune")
        return model, True
    except Exception:
        return model, False


def dry_forward(model, batch_size, max_len, device, dtype):
    # tiny warmup to build kernels; no synchronize()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        input_ids = torch.ones((batch_size, max_len), dtype=torch.long, device=device)
        attn = torch.ones((batch_size, max_len), dtype=torch.long, device=device)
        out = model(input_ids=input_ids, attention_mask=attn)
        _ = mean_pool(out.last_hidden_state, attn)


def autotune_batch(model, start, hi_limit, max_len, device, dtype):
    # Grow then binary-search. Be robust to RuntimeError and AssertionError from weird backends.
    lo, hi = start, start
    while hi <= hi_limit:
        try:
            dry_forward(model, hi, max_len, device, dtype)
            lo = hi
            hi *= 2
        except (RuntimeError, AssertionError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                pass
            torch.cuda.empty_cache()
            break
    left, right = lo, min(hi, hi_limit)
    best = lo
    while left <= right:
        mid = (left + right) // 2
        if mid == 0:
            break
        try:
            dry_forward(model, mid, max_len, device, dtype)
            best = mid
            left = mid + 1
        except (RuntimeError, AssertionError) as e:
            torch.cuda.empty_cache()
            right = mid - 1
    return best


class BucketedParquet(IterableDataset):
    def __init__(self, path, text_col, bucket_size=16384):
        super().__init__()
        self.path = path
        self.text_col = text_col
        self.bucket_size = bucket_size

    def __iter__(self):
        pf = pq.ParquetFile(self.path)
        wi = get_worker_info()
        num_workers = wi.num_workers if wi is not None else 1
        worker_id = wi.id if wi is not None else 0

        buf = []
        global_offset = 0
        for batch in pf.iter_batches(
            batch_size=self.bucket_size, columns=[self.text_col]
        ):
            col = batch.column(0).to_pylist()
            # shard by global index so workers see disjoint rows
            for i, t in enumerate(col):
                gidx = global_offset + i
                if (gidx % num_workers) != worker_id:
                    continue
                if not isinstance(t, str):
                    t = "" if t is None else str(t)
                buf.append((t, gidx))
                if len(buf) >= self.bucket_size:
                    buf.sort(key=lambda x: len(x[0]))
                    for b in buf:
                        yield b
                    buf.clear()
            global_offset += len(col)
        if buf:
            buf.sort(key=lambda x: len(x[0]))
            for b in buf:
                yield b


# =======================
# Load
# =======================
assert Path(PARQUET_PATH).exists(), f"File not found: {PARQUET_PATH}"
pf = pq.ParquetFile(PARQUET_PATH)
N_ROWS = pf.metadata.num_rows
if TEXT_COL is None:
    TEXT_COL = detect_text_column(pf)
print(f"[info] Using text column: {TEXT_COL}")
print(f"[info] Found ~{N_ROWS:,} rows in {PARQUET_PATH}")

# Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
model.eval()

# Try BetterTransformer first (safe)
model, used_bt = maybe_bettertransformer(model)

# Autotune (no compile here)
if AUTO_TUNE_BATCH:
    tuned = autotune_batch(
        model,
        start=BATCH,
        hi_limit=BATCH_LIMIT,
        max_len=MAX_LEN,
        device=DEVICE,
        dtype=DTYPE,
    )
    BATCH = max(1, int(tuned * 0.9))
print(f"[info] Using batch size: {BATCH}")

# (Optional) Compile AFTER autotune for speed; keep False if you saw earlier crash
used_compile = False
if USE_COMPILE:
    model, used_compile = maybe_compile(model)

# Hidden size for memmap
H = model.config.hidden_size
print(f"[info] Model hidden size: {H}")

# =======================
# Data + memmap
# =======================
dataset = BucketedParquet(PARQUET_PATH, TEXT_COL, bucket_size=16384)
loader = DataLoader(
    dataset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=PREFETCH if NUM_WORKERS > 0 else None,
    persistent_workers=True if NUM_WORKERS > 0 else False,
    collate_fn=lambda b: collate(b, tokenizer),
)

emb_map = np.memmap(OUT_EMB_PATH, dtype=np.float16, mode="w+", shape=(N_ROWS, H))

# =======================
# Run (CUDA events; single sync at end)
# =======================
t0 = time.perf_counter()
total_texts = 0
total_tokens = 0
step_ms = []
event_pairs = []
written = 0
skipped_runtime = 0

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

with torch.inference_mode(), torch.autocast(
    device_type="cuda", dtype=DTYPE if DEVICE == "cuda" else None
):
    for toks, indices in tqdm(
        loader, total=math.ceil(N_ROWS / BATCH), desc="Embedding"
    ):
        start_ev = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        end_ev = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        if start_ev is not None:
            start_ev.record()
        try:
            input_ids = toks["input_ids"].to(DEVICE, non_blocking=True)
            attn = toks["attention_mask"].to(DEVICE, non_blocking=True)

            out = model(input_ids=input_ids, attention_mask=attn)
            emb = mean_pool(out.last_hidden_state, attn)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            # metrics
            bsz = input_ids.size(0)
            total_tokens += int(attn.sum().item())
            total_texts += bsz

            # write by row indices to preserve alignment
            idx_np = np.asarray(indices, dtype=np.int64)
            emb_np = emb.to(torch.float16).detach().cpu().numpy()
            emb_map[idx_np, :] = emb_np

        except (RuntimeError, AssertionError) as e:
            skipped_runtime += len(indices)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            if end_ev is not None:
                end_ev.record()
                event_pairs.append((start_ev, end_ev))

# flush memmap
del emb_map
if torch.cuda.is_available():
    torch.cuda.synchronize()

# metrics
for s, e in event_pairs:
    if s is not None and e is not None:
        step_ms.append(s.elapsed_time(e))

T = time.perf_counter() - t0
avg_len = total_tokens / max(total_texts, 1)
texts_per_sec = total_texts / T if T > 0 else float("nan")
tokens_per_sec = total_tokens / T if T > 0 else float("nan")
p50 = float(np.percentile(step_ms, 50)) if step_ms else float("nan")
p95 = float(np.percentile(step_ms, 95)) if step_ms else float("nan")
peak_mem_gb = (
    (torch.cuda.max_memory_allocated() / (1024**3))
    if torch.cuda.is_available()
    else float("nan")
)
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

print("\n=== EMBEDDING RUN SUMMARY ===")
print(f"File: {PARQUET_PATH}  ->  Output: {OUT_EMB_PATH}")
print(f"Model: {MODEL_ID}")
print(f"GPU: {gpu_name}")
print(f"Device dtype (autocast): {str(DTYPE).replace('torch.', '')}")
print(
    f"Batch size: {BATCH} | Max length: {MAX_LEN} | Workers: {NUM_WORKERS} | Prefetch: {PREFETCH}"
)
print(f"Tokenizers parallelism: {os.environ.get('TOKENIZERS_PARALLELISM')}")
print(
    f"Optimizations -> BetterTransformer: {used_bt} | torch.compile: {used_compile} | SDPA enabled: True"
)
print(f"Hidden size: {H}")
print(
    f"Total texts written: {written:,} / {N_ROWS:,}  (skipped@runtime: {skipped_runtime:,})"
)
print(f"Total tokens (mask sum): {total_tokens:,}")
print(f"Avg effective tokens/text: {avg_len:.2f}")
print(f"Wall time: {T:.2f} s")
print(f"Throughput: {texts_per_sec:,.1f} texts/s | {tokens_per_sec:,.1f} tokens/s")
print(f"Per-batch latency: p50={p50:.2f} ms | p95={p95:.2f} ms")
print(f"Peak GPU memory: {peak_mem_gb:.2f} GB")
print("================================")

# =======================
# Meta file
# =======================
meta = {
    "model": MODEL_ID,
    "prefix": PREFIX,
    "dim": int(H),
    "normalized": True,
    "dtype": "float16",
    "max_len": int(MAX_LEN),
    "n_vectors": int(N_ROWS),
    "source": str(PARQUET_PATH),
    "text_col": TEXT_COL,
}
with open(OUT_META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"Saved meta to {OUT_META_PATH}")
