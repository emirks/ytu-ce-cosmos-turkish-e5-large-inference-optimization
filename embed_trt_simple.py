import os
import sys
from typing import List, Dict, Any, Optional

import numpy as np
import onnxruntime as ort  # type: ignore
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm  # type: ignore
from transformers import AutoTokenizer  # type: ignore


# =======================
# Minimal Config
# =======================
PARQUET_PATH: str = os.environ.get("CHUNKS_PARQUET", "chunks.parquet")
OUTPUT_PATH: str = os.environ.get("EMBEDDINGS_PARQUET", "chunk_embeddings.parquet")
MODEL_ID: str = os.environ.get("TOKENIZER_MODEL_ID", "ytu-ce-cosmos/turkish-e5-large")
ONNX_PATH: str = os.environ.get("ONNX_MODEL_PATH", "onnx_e5/model.onnx")
TEXT_COL: str = os.environ.get("TEXT_COL", "mdtext")
PREFIX: str = os.environ.get("E5_PREFIX", "passage: ")
MAX_LEN: int = int(os.environ.get("MAX_LEN", "512"))
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "512"))

# Columns we will carry through
META_COLUMNS: List[str] = [
    "id",
    "doc_id",
    "chunk_id",
    "start_char",
    "end_char",
    "num_tokens",
]


def build_session(onnx_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    providers: List[Any] = []

    # Try TensorRT first, then CUDA, then CPU
    try:
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./trt_cache",
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": "./trt_cache/timing.cache",
                    # Keeping it simple: let ORT create default profiles from first shapes
                },
            )
        )
    except Exception:
        pass

    try:
        providers.append(("CUDAExecutionProvider", {"enable_cuda_graph": False}))
    except Exception:
        pass

    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    print(f"Providers: {session.get_providers()}")
    return session


def map_input_names(session: ort.InferenceSession) -> Dict[str, Optional[str]]:
    ins = [i.name for i in session.get_inputs()]
    name_map: Dict[str, Optional[str]] = {
        "input_ids": next(
            (n for n in ins if "input_ids" in n), ins[0] if ins else None
        ),
        "attention_mask": next((n for n in ins if "attention_mask" in n), None),
        "token_type_ids": next((n for n in ins if "token_type_ids" in n), None),
    }
    if name_map["input_ids"] is None:
        raise RuntimeError("Unable to locate model input name for input_ids")
    return name_map


def mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    # last_hidden: (N, L, H) float16/float32
    # attention_mask: (N, L) int64/bool
    mask = attention_mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)  # (N, L, 1)
    summed = (last_hidden.astype(np.float32) * mask).sum(axis=1)  # (N, H)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)  # (N, 1)
    return summed / counts


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def run_batch(
    session: ort.InferenceSession,
    names: Dict[str, Optional[str]],
    tokenizer: AutoTokenizer,
    texts: List[str],
) -> np.ndarray:
    toks = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="np",
    )

    feed: Dict[str, Any] = {}
    feed[names["input_ids"]] = toks["input_ids"].astype(np.int64)
    if names["attention_mask"] is not None:
        feed[names["attention_mask"]] = toks["attention_mask"].astype(np.int64)
        attn = toks["attention_mask"].astype(np.int64)
    else:
        # If mask is missing, assume all tokens are valid
        attn = np.ones_like(toks["input_ids"], dtype=np.int64)

    if names["token_type_ids"] is not None and "token_type_ids" in toks:
        feed[names["token_type_ids"]] = toks["token_type_ids"].astype(np.int64)

    outputs = session.run(None, feed)
    last_hidden = outputs[0]  # (N, L, H)

    pooled = mean_pool(last_hidden, attn)
    normalized = l2_normalize(pooled, axis=1)
    return normalized.astype(np.float32)


def main() -> None:
    if not os.path.exists(PARQUET_PATH):
        raise SystemExit(f"File not found: {PARQUET_PATH}")
    if not os.path.exists(ONNX_PATH):
        raise SystemExit(f"ONNX model not found: {ONNX_PATH}")

    # Load
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    session = build_session(ONNX_PATH)
    names = map_input_names(session)

    # Prepare parquet reader/writer
    pf = pq.ParquetFile(PARQUET_PATH)

    # Define output schema (embedding as list<float>)
    emb_type = pa.list_(pa.float32())
    fields = [
        pa.field("id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("chunk_id", pa.int32()),
        pa.field("start_char", pa.int32()),
        pa.field("end_char", pa.int32()),
        pa.field("num_tokens", pa.int32()),
        pa.field("embedding", emb_type),
    ]
    schema = pa.schema(fields)

    writer = pq.ParquetWriter(OUTPUT_PATH, schema=schema)

    total_rows = pf.metadata.num_rows if pf.metadata is not None else None
    batch_iter = pf.iter_batches(
        batch_size=BATCH_SIZE, columns=META_COLUMNS + [TEXT_COL]
    )

    try:
        pbar = tqdm(total=total_rows, unit="row", dynamic_ncols=True, desc="Embedding")
    except Exception:
        pbar = None

    written = 0
    try:
        for rb in batch_iter:
            tbl = pa.Table.from_batches([rb])
            cols = {name: tbl[name].to_pylist() for name in tbl.column_names}

            texts_raw: List[Any] = cols.get(TEXT_COL, [])
            # Prepare metadata columns
            ids = [str(x) if x is not None else "" for x in cols.get("id", [])]
            doc_ids = [str(x) if x is not None else "" for x in cols.get("doc_id", [])]
            chunk_ids = [
                int(x) if x is not None else -1 for x in cols.get("chunk_id", [])
            ]
            start_chars = [
                int(x) if x is not None else -1 for x in cols.get("start_char", [])
            ]
            end_chars = [
                int(x) if x is not None else -1 for x in cols.get("end_char", [])
            ]
            num_tokens = [
                int(x) if x is not None else 0 for x in cols.get("num_tokens", [])
            ]

            # Filter invalid/empty texts
            texts: List[str] = []
            keep_idx: List[int] = []
            for i, t in enumerate(texts_raw):
                if t is None:
                    continue
                s = str(t).strip()
                if not s:
                    continue
                texts.append(PREFIX + s)
                keep_idx.append(i)

            if not texts:
                if pbar is not None:
                    pbar.update(len(texts_raw))
                continue

            emb = run_batch(session, names, tokenizer, texts)  # (M, H)

            # Build output lists with kept indices
            out_ids = [ids[i] for i in keep_idx]
            out_doc_ids = [doc_ids[i] for i in keep_idx]
            out_chunk_ids = [chunk_ids[i] for i in keep_idx]
            out_start = [start_chars[i] for i in keep_idx]
            out_end = [end_chars[i] for i in keep_idx]
            out_ntok = [num_tokens[i] for i in keep_idx]
            emb_lists = [row.tolist() for row in emb]

            out_tbl = pa.Table.from_pydict(
                {
                    "id": out_ids,
                    "doc_id": out_doc_ids,
                    "chunk_id": out_chunk_ids,
                    "start_char": out_start,
                    "end_char": out_end,
                    "num_tokens": out_ntok,
                    "embedding": emb_lists,
                },
                schema=schema,
            )
            writer.write_table(out_tbl)
            written += out_tbl.num_rows

            if pbar is not None:
                pbar.update(len(texts_raw))
    finally:
        writer.close()
        if pbar is not None:
            pbar.close()

    print(f"Wrote {written} embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
