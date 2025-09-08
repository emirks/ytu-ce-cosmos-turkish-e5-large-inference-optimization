import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss  # pip install faiss-gpu-cu12 (or matching CUDA)
from tqdm import tqdm

# =======================
# Hardcoded config
# =======================
EMBEDDINGS_ROOT = "/content/embeddings"
OUTPUT_INDEX_PATH = "/content/embeddings/faiss_flat_cosine.index"
OUTPUT_META_PATH = "/content/embeddings/faiss_flat_cosine.meta.json"

# GPU/Index settings
USE_GPU = True
GPU_DEVICE = 0  # change if needed
GPU_USE_FLOAT16_STORAGE = True  # halves GPU memory for vectors

# IO/streaming settings
ADD_BATCH_ROWS = 131072  # tune based on GPU/CPU RAM; ~128k rows per add
PRINT_EVERY_BATCHES = 10

# Validation/safety
REQUIRE_NORMALIZED = True  # embeddings should be L2-normalized for cosine/IP


# =======================
# Helpers
# =======================


def _list_shard_dirs(root: str) -> List[Path]:
    # Expect folders like docs_0m_1m, docs_1m_2m, ... under EMBEDDINGS_ROOT
    root_path = Path(root)
    candidates = [p for p in root_path.glob("docs_*_*") if p.is_dir()]

    # Sort by start million if possible (extract number before the first 'm')
    def sort_key(p: Path) -> Tuple[int, str]:
        m = re.search(r"docs_(\d+)m_", p.name)
        return (int(m.group(1)) if m else 0, p.name)

    return sorted(candidates, key=sort_key)


def _find_files_for_shard(shard_dir: Path) -> Tuple[Path, Path, Path]:
    # Return (embeddings_fp16_npy, chunks_parquet, meta_json)
    emb = None
    chunks = None
    meta = None
    for f in shard_dir.iterdir():
        name = f.name
        if name.startswith("embeddings_") and name.endswith(".fp16.npy"):
            emb = f
        elif name.startswith("chunks_") and name.endswith(".parquet"):
            chunks = f
        elif name.startswith("meta_") and name.endswith(".json"):
            meta = f
    if emb is None:
        # Fallback: maybe only one embeddings file with different suffix
        matches = list(shard_dir.glob("embeddings_*.npy"))
        if matches:
            emb = matches[0]
    if meta is None:
        # meta is optional but recommended by the embedding script; continue without it
        pass
    if emb is None or chunks is None:
        raise FileNotFoundError(
            f"Missing required files in {shard_dir}: emb={emb}, chunks={chunks}"
        )
    return emb, chunks, (meta if meta is not None else Path(""))


def _read_meta_dim_n(meta_path: Path, fallback_chunks: Path) -> Tuple[int, int, bool]:
    dim = None
    n_vectors = None
    normalized = None
    if meta_path and meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            dim = int(m.get("dim")) if "dim" in m else None
            n_vectors = int(m.get("n_vectors")) if "n_vectors" in m else None
            normalized = bool(m.get("normalized")) if "normalized" in m else None
        except Exception:
            pass
    if n_vectors is None or dim is None:
        # Fallback: infer via parquet rows and load a tiny slice of embeddings
        import pyarrow.parquet as pq  # local import to avoid hard dep in environments without pyarrow

        pf = pq.ParquetFile(str(fallback_chunks))
        n_vectors = pf.metadata.num_rows
        # Try to infer dim by peeking the memmap later; here we just return placeholders
    if normalized is None:
        normalized = False
    return dim if dim is not None else -1, n_vectors, normalized


def _infer_dim_from_memmap(path: Path, n_vectors: int) -> int:
    # The embeddings were written via numpy.memmap, not np.save, so reopen similarly
    # We need to try common dims if meta is missing. We'll probe a few likely dims.
    likely_dims = [768, 1024, 512, 384, 256]
    for d in likely_dims:
        try:
            _ = np.memmap(str(path), dtype=np.float16, mode="r", shape=(n_vectors, d))
            return d
        except Exception:
            continue
    raise RuntimeError(
        "Unable to infer embedding dimensionality. Provide meta_*.json with 'dim'."
    )


# =======================
# Build index
# =======================


def build_index():
    shard_dirs = _list_shard_dirs(EMBEDDINGS_ROOT)
    if not shard_dirs:
        # Allow single-folder setup (e.g., only one docs_0m_1m exists or root contains files directly)
        single = Path(EMBEDDINGS_ROOT)
        if single.is_dir():
            shard_dirs = [single]
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories found under {EMBEDDINGS_ROOT}")

    shard_infos: List[Dict] = []
    total_rows = 0
    dim_global = None

    print(f"[info] Scanning shards under: {EMBEDDINGS_ROOT}")
    for sd in shard_dirs:
        try:
            emb_path, chunks_path, meta_path = _find_files_for_shard(sd)
        except FileNotFoundError:
            continue
        dim, n_rows, was_normalized = _read_meta_dim_n(meta_path, chunks_path)
        if dim < 0:
            dim = _infer_dim_from_memmap(emb_path, n_rows)
        shard_infos.append(
            {
                "dir": str(sd),
                "embeddings": str(emb_path),
                "chunks": str(chunks_path),
                "meta": str(meta_path) if meta_path else "",
                "dim": dim,
                "n_rows": n_rows,
                "normalized": was_normalized,
            }
        )
        total_rows += n_rows
        if dim_global is None:
            dim_global = dim
        elif dim_global != dim:
            raise ValueError(
                f"Dimension mismatch across shards: {dim_global} vs {dim} in {sd}"
            )

    if not shard_infos:
        raise FileNotFoundError("No valid shards with embeddings/chunks found.")

    assert dim_global is not None

    if REQUIRE_NORMALIZED:
        any_norm = any(si["normalized"] for si in shard_infos)
        if not any_norm:
            print(
                "[warn] Embeddings do not appear to be normalized in meta. Proceeding regardless."
            )

    print(
        f"[info] Total shards: {len(shard_infos)} | Dim: {dim_global} | Total vectors: {total_rows:,}"
    )

    # Create FAISS index (Flat IP ≈ cosine when vectors are L2-normalized)
    metric = faiss.METRIC_INNER_PRODUCT

    if USE_GPU:
        num_gpus = faiss.get_num_gpus()
        if num_gpus <= GPU_DEVICE:
            print(
                f"[warn] Requested GPU device {GPU_DEVICE} not available. Falling back to CPU."
            )
            use_gpu = False
        else:
            use_gpu = True
    else:
        use_gpu = False

    if use_gpu:
        print(
            f"[info] Creating GPU FlatIP index on device {GPU_DEVICE} (float16_storage={GPU_USE_FLOAT16_STORAGE})"
        )
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = GPU_DEVICE
        cfg.useFloat16 = GPU_USE_FLOAT16_STORAGE
        gpu_index = faiss.GpuIndexFlatIP(res, dim_global, cfg)
        index = gpu_index
    else:
        print("[info] Creating CPU FlatIP index")
        index = faiss.IndexFlatIP(dim_global)

    # Stream-add vectors shard by shard
    t0 = time.perf_counter()
    rows_added = 0
    batch_counter = 0

    for si in shard_infos:
        emb_path = Path(si["embeddings"])
        n_rows = int(si["n_rows"])
        dim = int(si["dim"])
        print(f"[info] Adding shard: {emb_path.parent.name} | rows={n_rows:,}")

        mm = np.memmap(str(emb_path), dtype=np.float16, mode="r", shape=(n_rows, dim))
        with tqdm(total=n_rows, desc=f"Add {emb_path.parent.name}", unit="vec") as pbar:
            for start in range(0, n_rows, ADD_BATCH_ROWS):
                end = min(start + ADD_BATCH_ROWS, n_rows)
                batch_fp16 = mm[start:end]
                # Convert to float32 for FAISS input
                batch = batch_fp16.astype(np.float32, copy=False)
                index.add(batch)
                rows_added += end - start
                batch_counter += 1
                pbar.update(end - start)
                if (batch_counter % PRINT_EVERY_BATCHES) == 0:
                    elapsed = time.perf_counter() - t0
                    rate = rows_added / max(elapsed, 1e-6)
                    print(
                        f"[info] Added {rows_added:,}/{total_rows:,} vectors | {rate:,.0f} vec/s"
                    )

    elapsed_total = time.perf_counter() - t0
    rate_total = rows_added / max(elapsed_total, 1e-6)
    print(
        f"[done] All vectors added: {rows_added:,} in {elapsed_total:,.1f}s ({rate_total:,.0f} vec/s)"
    )

    # Move to CPU if built on GPU before saving
    if use_gpu:
        print("[info] Copying index from GPU to CPU for serialization ...")
        index = faiss.index_gpu_to_cpu(index)

    # Save index
    print(f"[info] Writing index to: {OUTPUT_INDEX_PATH}")
    Path(OUTPUT_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, OUTPUT_INDEX_PATH)

    # Write meta JSON for later lookups
    meta = {
        "index_path": OUTPUT_INDEX_PATH,
        "index_type": "FlatIP",
        "metric": "cosine_via_inner_product",
        "dim": int(dim_global),
        "total_vectors": int(rows_added),
        "normalized_expected": REQUIRE_NORMALIZED,
        "gpu_used": bool(use_gpu),
        "gpu_device": int(GPU_DEVICE) if use_gpu else None,
        "gpu_float16_storage": bool(GPU_USE_FLOAT16_STORAGE) if use_gpu else None,
        "shards": [
            {
                "dir": si["dir"],
                "embeddings": si["embeddings"],
                "chunks": si["chunks"],
                "meta": si["meta"],
                "n_rows": int(si["n_rows"]),
                "dim": int(si["dim"]),
                "normalized": bool(si["normalized"]),
            }
            for si in shard_infos
        ],
        "notes": "Vector IDs in the FAISS index correspond to the row order across shards, concatenated in the scan order above. Rows align 1:1 with the corresponding Parquet rows in each shard.",
    }
    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[info] Wrote meta to: {OUTPUT_META_PATH}")


if __name__ == "__main__":
    build_index()
