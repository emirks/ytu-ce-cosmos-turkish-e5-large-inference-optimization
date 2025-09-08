#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# ---- HARD-CODED SETTINGS ----
INDEX_PATH = "/content/embeddings/docs_0m_1m/index_ivfpq.faiss"
MODEL_ID = "ytu-ce-cosmos/turkish-e5-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
NPROBE = 64  # only used for IVF/IVFPQ
USE_GPU = True  # set False to run search on CPU
GPU_ID = 0
MAX_LEN = 512
QUERY_PREFIX = (
    "query: "  # e5 convention: use "query: " for queries, "passage: " for docs
)
TEXT_SNIPPET_LEN = 300
# -----------------------------


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-6)
    return summed / counts


def embed_query(text: str, tokenizer, model) -> np.ndarray:
    text = text.strip()
    if not text:
        raise ValueError("Empty query text")
    toks = tokenizer(
        f"{QUERY_PREFIX}{text}",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16 if DEVICE == "cuda" else None
    ):
        input_ids = toks["input_ids"].to(DEVICE)
        attn = toks["attention_mask"].to(DEVICE)
        out = model(input_ids=input_ids, attention_mask=attn)
        emb = mean_pool(out.last_hidden_state, attn)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        emb_np = emb.to(torch.float32).detach().cpu().numpy()
        return emb_np


def load_index(index_path: str):
    index = faiss.read_index(index_path)
    return index


def maybe_to_gpu(index):
    if not USE_GPU:
        return index, None
    if faiss.get_num_gpus() <= 0:
        return index, None
    if not (0 <= GPU_ID < faiss.get_num_gpus()):
        return index, None
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    gpu_index = faiss.index_cpu_to_gpu(res, GPU_ID, index, co)
    return gpu_index, res


def load_index_meta(index_path: str):
    meta_path = index_path + ".meta.json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_rows_by_indices(parquet_path: str, indices, columns=None):
    if pq is None:
        return {}
    pf = pq.ParquetFile(parquet_path)
    targets = set(int(i) for i in indices)
    found = {}
    cols = (
        columns
        if columns
        else [
            "doc_id",
            "chunk_id",
            "start_char",
            "end_char",
            "num_tokens",
            "mdtext",
            "source_path",
            "id",
            "doc_checksum",
        ]
    )
    global_offset = 0
    for batch in pf.iter_batches(batch_size=16384, columns=cols):
        batch_len = len(batch)
        # Convert columns once per batch for quick row access
        col_arrays = [batch.column(i).to_pylist() for i in range(len(cols))]
        for i in range(batch_len):
            gidx = global_offset + i
            if gidx in targets:
                row = {cols[c]: col_arrays[c][i] for c in range(len(cols))}
                found[gidx] = row
                if len(found) == len(targets):
                    return found
        global_offset += batch_len
    return found


def main():
    # Get query text from CLI or prompt
    query_text = " ".join(sys.argv[1:]).strip()
    if not query_text:
        query_text = input("Enter query: ").strip()
    if not query_text:
        print("No query provided.")
        sys.exit(1)

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if DEVICE == "cuda" else None
    ).to(DEVICE)
    model.eval()

    # Embed query
    q = embed_query(query_text, tokenizer, model)  # shape (1, d)

    # Load index
    index = load_index(INDEX_PATH)

    # Set nprobe if IVF family
    if hasattr(index, "nprobe"):
        try:
            index.nprobe = NPROBE
        except Exception:
            pass

    # Optionally move to GPU
    index_to_use = index
    gpu_res = None
    if USE_GPU:
        index_to_use, gpu_res = maybe_to_gpu(index)
        if hasattr(index_to_use, "nprobe"):
            try:
                index_to_use.nprobe = NPROBE
            except Exception:
                pass

    # Search
    D, I = index_to_use.search(q.astype("float32"), TOP_K)

    # Load meta to find parquet path
    meta = load_index_meta(INDEX_PATH)
    chunks_parquet = meta.get("chunks_parquet")

    rows = {}
    if chunks_parquet and pq is not None:
        rows = fetch_rows_by_indices(chunks_parquet, I[0])

    # Print results
    print("\n=== Search results ===")
    print(f"Query: {query_text}")
    for rank, (vid, score) in enumerate(zip(I[0], D[0]), start=1):
        vid_int = int(vid)
        print(f"{rank:2d}. id={vid_int}\tscore={float(score):.6f}")
        r = rows.get(vid_int)
        if r is not None:
            snippet = r.get("mdtext") or ""
            if len(snippet) > TEXT_SNIPPET_LEN:
                snippet = snippet[:TEXT_SNIPPET_LEN] + "..."
            print(
                f"    doc_id={r.get('doc_id')} chunk_id={r.get('chunk_id')} char=[{r.get('start_char')},{r.get('end_char')}] tokens={r.get('num_tokens')}"
            )
            print(f"    text: {snippet}")
        else:
            if not chunks_parquet:
                print(
                    "    (no parquet metadata available; install pyarrow or check index meta)"
                )
            else:
                print("    (row not found in parquet)")


if __name__ == "__main__":
    main()
