import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm

# Hardcoded root directory to scan (change if needed)
ROOT_DIR = "."


def find_chunk_embedding_pairs(root: str) -> List[Tuple[Path, Optional[Path]]]:
    """
    Find chunk.parquet files and their corresponding embedding.npy files.
    Returns list of (chunk_path, embedding_path) tuples.
    """
    root_path = Path(root)
    chunk_files = sorted(root_path.rglob("chunks_*.parquet"))

    pairs: List[Tuple[Path, Optional[Path]]] = []

    for chunk_path in chunk_files:
        # Extract the million range from chunk filename
        # e.g., chunks_0m_1m.parquet -> embeddings_0m_1m.fp16.npy
        chunk_name = chunk_path.stem  # "chunks_0m_1m"
        if chunk_name.startswith("chunks_"):
            embedding_name = chunk_name.replace("chunks_", "embeddings_") + ".fp16.npy"
            embedding_path = chunk_path.parent / embedding_name

            if embedding_path.exists():
                pairs.append((chunk_path, embedding_path))
            else:
                pairs.append((chunk_path, None))

    return pairs


def get_parquet_row_count(parquet_path: Path) -> int:
    """Get row count from parquet file metadata."""
    pf = pq.ParquetFile(str(parquet_path))
    return pf.metadata.num_rows


def get_numpy_row_count(numpy_path: Path) -> int:
    """Get row count from numpy file."""
    try:
        # Load just the shape without loading the full array
        with open(numpy_path, "rb") as f:
            # Skip header to get shape info
            f.seek(128)  # Skip numpy header
            shape = np.lib.format.read_array_header_1_0(f)[0]
            return shape[0] if len(shape) > 0 else 0
    except Exception as e:
        print(f"[error] Failed to read numpy shape from {numpy_path}: {e}")
        return -1


def validate_pair(
    chunk_path: Path, embedding_path: Optional[Path]
) -> Dict[str, object]:
    """Validate a chunk/embedding pair."""
    chunk_rows = get_parquet_row_count(chunk_path)

    result = {
        "chunk_path": str(chunk_path),
        "chunk_rows": chunk_rows,
        "embedding_path": str(embedding_path) if embedding_path else None,
        "embedding_rows": None,
        "match": False,
        "status": "missing_embedding" if embedding_path is None else "unknown",
    }

    if embedding_path is None:
        return result

    embedding_rows = get_numpy_row_count(embedding_path)
    result["embedding_rows"] = embedding_rows

    if embedding_rows == -1:
        result["status"] = "embedding_read_error"
    elif chunk_rows == embedding_rows:
        result["status"] = "valid"
        result["match"] = True
    else:
        result["status"] = "mismatch"
        result["match"] = False

    return result


def main():
    print(f"[info] Scanning for chunk/embedding pairs under {ROOT_DIR}")

    pairs = find_chunk_embedding_pairs(ROOT_DIR)
    if not pairs:
        print(f"No chunk files found under {ROOT_DIR}")
        return

    print(f"[info] Found {len(pairs)} chunk files")

    results: List[Dict[str, object]] = []
    valid_count = 0
    mismatch_count = 0
    missing_count = 0
    error_count = 0

    for chunk_path, embedding_path in tqdm(pairs, desc="Validating pairs"):
        result = validate_pair(chunk_path, embedding_path)
        results.append(result)

        # Live logging
        status = result["status"]
        chunk_rows = result["chunk_rows"]

        if status == "valid":
            valid_count += 1
            print(f"[✓] {chunk_path.name} | rows={chunk_rows:,} | ✓")
        elif status == "mismatch":
            mismatch_count += 1
            emb_rows = result["embedding_rows"]
            print(
                f"[✗] {chunk_path.name} | chunk={chunk_rows:,} | embedding={emb_rows:,} | MISMATCH"
            )
        elif status == "missing_embedding":
            missing_count += 1
            print(f"[?] {chunk_path.name} | rows={chunk_rows:,} | missing embedding")
        elif status == "embedding_read_error":
            error_count += 1
            print(f"[!] {chunk_path.name} | rows={chunk_rows:,} | embedding read error")

    # Summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Total chunk files: {len(pairs)}")
    print(f"Valid pairs: {valid_count}")
    print(f"Row count mismatches: {mismatch_count}")
    print(f"Missing embeddings: {missing_count}")
    print(f"Embedding read errors: {error_count}")

    if mismatch_count > 0:
        print("\n[WARNING] Found row count mismatches!")
        print("Files with mismatches:")
        for result in results:
            if result["status"] == "mismatch":
                chunk_path = result["chunk_path"]
                chunk_rows = result["chunk_rows"]
                emb_rows = result["embedding_rows"]
                print(f"  {chunk_path}: chunk={chunk_rows:,}, embedding={emb_rows:,}")

    if missing_count > 0:
        print("\n[INFO] Files missing embeddings:")
        for result in results:
            if result["status"] == "missing_embedding":
                chunk_path = result["chunk_path"]
                print(f"  {chunk_path}")

    print("==========================")


if __name__ == "__main__":
    main()
