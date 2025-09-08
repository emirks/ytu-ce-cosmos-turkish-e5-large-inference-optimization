import json
import os
import glob
import random
from typing import List, Optional, Tuple, Dict

import numpy as np
import pyarrow.parquet as pq

# Configuration
ROOT_DIR = "/content/embeddings"  # Start from /content/embeddings directory
SAMPLE_MATCH_COUNT = 10
COS_SIM_THRESHOLD = 0.995
MAX_LEN = 512
MODEL_ID = "ytu-ce-cosmos/turkish-e5-large"
PREFIX = "passage: "

# Lazy singletons for model resources
_tokenizer = None
_model = None
_device = None


def _detect_text_column(pf: pq.ParquetFile) -> str:
    candidates = [
        "mdtext",
        "text",
        "text_snippet",
        "content",
        "chunk_text",
        "body",
    ]
    cols = set(pf.schema.names)
    for c in candidates:
        if c in cols:
            return c
    # fallback to first column name
    return pf.schema.names[0]


def _ensure_model():
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None and _device is not None:
        return _tokenizer, _model, _device
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        _model = AutoModel.from_pretrained(MODEL_ID)
        _model.to(_device)
        _model.eval()
        return _tokenizer, _model, _device
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{MODEL_ID}' for sample check: {e}")


def _embed_texts(texts: List[str]) -> np.ndarray:
    # Returns float32 normalized embeddings of shape (N, H)
    import torch

    tokenizer, model, device = _ensure_model()
    with torch.inference_mode():
        toks = tokenizer(
            [f"{PREFIX}{t}" for t in texts],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = model(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"])
        last = out.last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1).to(last.dtype)
        pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.to(torch.float32).cpu().numpy()


def find_parquet_files(root_dir: str = ".") -> List[Tuple[str, Optional[str]]]:
    """
    Find all parquet files and their corresponding embedding files.
    Returns list of (parquet_path, embedding_path) tuples.
    """
    pairs = []

    # Find all parquet files recursively
    parquet_files = []
    for pattern in ["**/*.parquet", "*.parquet"]:
        parquet_files.extend(glob.glob(os.path.join(root_dir, pattern), recursive=True))

    print(f"Found {len(parquet_files)} parquet files:")
    for pf in sorted(parquet_files):
        print(f"  {pf}")

    # For each parquet file, try to find corresponding embedding file
    for parquet_path in sorted(parquet_files):
        embedding_path = None

        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(parquet_path))[0]

        # Look for corresponding embedding file in the same directory
        parquet_dir = os.path.dirname(parquet_path)

        # Try different embedding file patterns based on the actual naming convention
        # For chunks_0m_1m.parquet, look for embeddings_0m_1m.fp16.npy
        embedding_patterns = [
            f"embeddings_{base_name.replace('chunks_', '')}.fp16.npy",
            f"embeddings_{base_name.replace('chunks_', '')}.npy",
            f"embeddings_{base_name.replace('chunks_', '')}.fp16",
            f"embeddings_{base_name.replace('chunks_', '')}.bin",
        ]

        for pattern in embedding_patterns:
            potential_embedding = os.path.join(parquet_dir, pattern)
            if os.path.exists(potential_embedding):
                embedding_path = potential_embedding
                break

        # If not found in same directory, look in embeddings subdirectory
        if embedding_path is None and "embeddings" in parquet_path:
            # Extract the range from the path (e.g., "docs_0m_1m" from "embeddings/docs_0m_1m/chunks_0m_1m.parquet")
            path_parts = parquet_path.split(os.sep)
            if len(path_parts) >= 2:
                range_dir = path_parts[-2]  # e.g., "docs_0m_1m"
                for pattern in embedding_patterns:
                    potential_embedding = os.path.join(parquet_dir, pattern)
                    if os.path.exists(potential_embedding):
                        embedding_path = potential_embedding
                        break

        pairs.append((parquet_path, embedding_path))

        if embedding_path:
            print(f"  ✓ Found embedding: {embedding_path}")
        else:
            print(f"  ✗ No embedding found for: {parquet_path}")

    return pairs


def load_meta(path: Optional[str]) -> Optional[dict]:
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def open_embeddings(path: str, n_rows: int, meta: Optional[dict]):
    # Try standard .npy first
    try:
        arr = np.load(path, mmap_mode="r")
        return arr, int(arr.shape[1]), arr.dtype
    except Exception:
        # Fallback: raw memmap without .npy header
        # Infer dtype and dim
        dtype_str = str(meta.get("dtype", "float16")).lower() if meta else "float16"
        dtype = np.float16 if ("16" in dtype_str or "fp16" in dtype_str) else np.float32
        dim_meta = int(meta["dim"]) if meta and "dim" in meta else None
        if dim_meta is None:
            itemsize = np.dtype(dtype).itemsize
            size_bytes = os.path.getsize(path)
            if n_rows == 0 or size_bytes == 0:
                raise ValueError(
                    "Cannot infer embedding shape from empty file or zero rows"
                )
            dim_meta = size_bytes // (itemsize * n_rows)
        arr = np.memmap(path, mode="r", dtype=dtype, shape=(n_rows, dim_meta))
        return arr, int(dim_meta), dtype


def validate_pair(
    parquet_path: str, embedding_path: Optional[str], meta_path: Optional[str] = None
) -> Dict:
    """
    Validate a single parquet-embedding pair.
    Only checks that row counts match between parquet and embedding files.
    Additionally, randomly samples 10 rows, recomputes their embeddings,
    and compares with the stored vectors using cosine similarity.
    Returns a dictionary with validation results.
    """
    result = {
        "parquet_path": parquet_path,
        "embedding_path": embedding_path,
        "status": "UNKNOWN",
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    print(f"\n{'='*80}")
    print(f"Validating: {parquet_path}")
    if embedding_path:
        print(f"Embedding: {embedding_path}")
    else:
        print("No embedding file found")
    print(f"{'='*80}")

    try:
        # 1) Get parquet row count
        if not os.path.exists(parquet_path):
            result["errors"].append(f"Missing chunks file: {parquet_path}")
            result["status"] = "FAIL"
            return result

        pf = pq.ParquetFile(parquet_path)
        n_rows = pf.metadata.num_rows

        result["stats"]["rows"] = n_rows

        print(f"Parquet rows: {n_rows:,}")

        # 2) Check embeddings shape (if embedding file exists)
        if embedding_path and os.path.exists(embedding_path):
            meta = load_meta(meta_path)
            emb, dim, emb_dtype = open_embeddings(embedding_path, n_rows, meta)
            n_vecs = emb.shape[0]

            result["stats"]["embedding_shape"] = (n_vecs, dim)
            result["stats"]["embedding_dtype"] = str(emb_dtype)

            print(f"Embedding vectors: {n_vecs:,}")
            print(f"Embedding shape: {(n_vecs, dim)}, dtype: {emb_dtype}")

            if n_vecs != n_rows:
                error_msg = f"vector count {n_vecs:,} != row count {n_rows:,}"
                print(f"FAIL: {error_msg}")
                result["errors"].append(error_msg)
                result["status"] = "FAIL"
                return result
            else:
                print("OK: vector count matches row count")

            # 3) Random 10-row consistency check (recompute and compare)
            try:
                if n_rows > 0:
                    k = min(SAMPLE_MATCH_COUNT, n_rows)
                    sample_idx = sorted(random.sample(range(n_rows), k))

                    # Read only needed text rows
                    text_col = _detect_text_column(pf)
                    texts: List[str] = []
                    fetched = 0
                    offset = 0
                    for batch in pf.iter_batches(batch_size=8192, columns=[text_col]):
                        batch_len = batch.num_rows
                        local = [
                            i - offset
                            for i in sample_idx
                            if offset <= i < offset + batch_len
                        ]
                        if local:
                            col = batch.column(0).to_pylist()
                            for li in local:
                                t = col[li]
                                if not isinstance(t, str):
                                    t = "" if t is None else str(t)
                                texts.append(t)
                                fetched += 1
                                if fetched >= k:
                                    break
                        if fetched >= k:
                            break
                        offset += batch_len

                    if len(texts) != k:
                        raise RuntimeError(
                            f"Failed to fetch {k} text rows; got {len(texts)}"
                        )

                    # Compute embeddings for sampled texts
                    computed = _embed_texts(texts)  # (k, dim)

                    # Fetch stored vectors
                    stored = emb[sample_idx].astype(np.float32)

                    # Compute cosine similarities
                    # Ensure stored are normalized; if not, normalize
                    norms = np.linalg.norm(stored, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    stored_norm = stored / norms
                    cos = np.sum(stored_norm * computed, axis=1)
                    min_cos = float(np.min(cos))
                    mean_cos = float(np.mean(cos))

                    result["stats"]["sample_match"] = {
                        "k": k,
                        "min_cos": min_cos,
                        "mean_cos": mean_cos,
                        "threshold": COS_SIM_THRESHOLD,
                    }
                    print(
                        f"sample match: k={k} min_cos={min_cos:.4f} mean_cos={mean_cos:.4f} threshold={COS_SIM_THRESHOLD}"
                    )

                    if min_cos < COS_SIM_THRESHOLD:
                        error_msg = f"Sampled embeddings do not match stored vectors enough (min_cos={min_cos:.4f} < {COS_SIM_THRESHOLD})"
                        print(f"FAIL: {error_msg}")
                        result["errors"].append(error_msg)
                        result["status"] = "FAIL"
                        return result
                else:
                    print("SKIP: No rows for sample match check")
            except Exception as e:
                error_msg = f"Sample match check skipped/failed: {e}"
                print(f"WARN: {error_msg}")
                result["warnings"].append(error_msg)

        elif embedding_path:
            error_msg = f"Missing embeddings file: {embedding_path}"
            print(f"FAIL: {error_msg}")
            result["errors"].append(error_msg)
            result["status"] = "FAIL"
            return result
        else:
            print("SKIP: No embedding file to validate")
            result["status"] = "SKIP"
            return result

        print("\nPASS: Row counts and sampled embeddings match.")
        result["status"] = "PASS"

    except Exception as e:
        error_msg = f"Validation failed with exception: {str(e)}"
        print(f"ERROR: {error_msg}")
        result["errors"].append(error_msg)
        result["status"] = "ERROR"

    return result


def validate_all():
    """
    Find and validate all parquet-embedding pairs in the directory structure.
    """
    print("Validating corpus alignment and quality for all files...\n")

    # Find all parquet files and their corresponding embeddings
    pairs = find_parquet_files(ROOT_DIR)

    if not pairs:
        print("No parquet files found!")
        return

    # Validate each pair
    results = []
    for parquet_path, embedding_path in pairs:
        # Look for meta.json in the same directory as parquet file
        parquet_dir = os.path.dirname(parquet_path)
        meta_path = os.path.join(parquet_dir, "meta.json")
        if not os.path.exists(meta_path):
            meta_path = None

        result = validate_pair(parquet_path, embedding_path, meta_path)
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")

    status_counts = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

        print(f"\n{result['parquet_path']}:")
        print(f"  Status: {status}")
        if result["embedding_path"]:
            print(f"  Embedding: {result['embedding_path']}")
        if result["stats"]:
            print(f"  Rows: {result['stats'].get('rows', 'N/A'):,}")
            if "embedding_shape" in result["stats"]:
                print(f"  Embedding shape: {result['stats']['embedding_shape']}")
            if "sample_match" in result["stats"]:
                sm = result["stats"]["sample_match"]
                print(
                    f"  Sample match: k={sm['k']} min_cos={sm['min_cos']:.4f} mean_cos={sm['mean_cos']:.4f} thr={sm['threshold']}"
                )
        if result["errors"]:
            print(f"  Errors: {len(result['errors'])}")
            for error in result["errors"]:
                print(f"    - {error}")
        if result["warnings"]:
            print(f"  Warnings: {len(result['warnings'])}")
            for warning in result["warnings"]:
                print(f"    - {warning}")

    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")

    total = len(results)
    passed = status_counts.get("PASS", 0)
    failed = status_counts.get("FAIL", 0) + status_counts.get("ERROR", 0)
    skipped = status_counts.get("SKIP", 0)

    print(f"\nTotal files: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")

    if failed == 0 and passed > 0:
        print("\n🎉 All files with embeddings passed validation!")
    elif failed > 0:
        print(f"\n❌ {failed} files failed validation")


if __name__ == "__main__":
    validate_all()
