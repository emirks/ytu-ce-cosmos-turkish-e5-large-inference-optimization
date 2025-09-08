import os
import sys
import math
from pathlib import Path
from typing import Dict, Optional, List

import pyarrow.parquet as pq
from tqdm import tqdm

# Hardcoded root directory to scan (change if needed)
ROOT_DIR = "."

# Recognized column names to count distinct documents
DOC_ID_CANDIDATES = [
    "doc_id",  # our chunked output
    "documentid",  # source dataset
    "document_id",
    "docid",
]


def find_parquet_files(root: str) -> List[Path]:
    root_path = Path(root)
    return sorted(root_path.rglob("*.parquet"))


def pick_doc_id_column(pf: pq.ParquetFile) -> Optional[str]:
    available = set(pf.schema.names)
    for name in DOC_ID_CANDIDATES:
        if name in available:
            return name
    return None


def count_distinct_in_column(
    parquet_path: Path, column_name: str, batch_size: int = 1 << 18
) -> int:
    pf = pq.ParquetFile(str(parquet_path))
    unique_values = set()
    with tqdm(
        total=pf.metadata.num_rows,
        desc=f"[distinct:{column_name}] {parquet_path.name}",
        unit="rows",
        leave=False,
    ) as pbar:
        for batch in pf.iter_batches(batch_size=batch_size, columns=[column_name]):
            col = batch.column(0).to_pylist()
            for v in col:
                if v is None:
                    continue
                unique_values.add(v)
            pbar.update(len(col))
    return len(unique_values)


def analyze_parquet(parquet_path: Path) -> Dict[str, object]:
    pf = pq.ParquetFile(str(parquet_path))
    num_rows = pf.metadata.num_rows
    cols = list(pf.schema.names)
    doc_col = pick_doc_id_column(pf)

    result: Dict[str, object] = {
        "path": str(parquet_path),
        "file": parquet_path.name,
        "rows": int(num_rows),
        "columns": cols,
        "doc_col": doc_col or "",
        "distinct_docs": None,
        "rows_per_doc_avg": None,
    }

    if doc_col is not None:
        distinct_docs = count_distinct_in_column(parquet_path, doc_col)
        result["distinct_docs"] = int(distinct_docs)
        if distinct_docs > 0:
            result["rows_per_doc_avg"] = float(num_rows / distinct_docs)

    return result


def main():
    parquet_files = find_parquet_files(ROOT_DIR)
    if not parquet_files:
        print(f"No parquet files found under {ROOT_DIR}")
        return

    print(f"[info] Found {len(parquet_files)} parquet files under {ROOT_DIR}")

    results: List[Dict[str, object]] = []

    for p in tqdm(parquet_files, desc="Scanning files"):
        try:
            res = analyze_parquet(p)
            results.append(res)
            # live log per file
            rows = res["rows"]
            doc_col = res["doc_col"] or "-"
            dd = res["distinct_docs"]
            rpd = res["rows_per_doc_avg"]
            msg = f"[ok] {p} | rows={rows:,} | doc_col={doc_col}"
            if dd is not None:
                msg += f" | distinct_docs={dd:,}"
            if rpd is not None:
                msg += f" | rows_per_doc_avg={rpd:.2f}"
            print(msg)
        except Exception as e:
            print(f"[warn] Failed to analyze {p}: {e}")

    # summary
    total_rows = sum(int(r["rows"]) for r in results)
    total_files = len(results)

    print("\n=== PARQUET SUMMARY ===")
    print(f"Files analyzed: {total_files}")
    print(f"Total rows: {total_rows:,}")
    # Summarize by doc column presence
    with_doc = [r for r in results if r.get("distinct_docs") is not None]
    if with_doc:
        total_docs = sum(int(r["distinct_docs"]) for r in with_doc)
        print(f"Files with doc column: {len(with_doc)}")
        print(
            f"Sum of distinct docs (per-file, not deduped across files): {total_docs:,}"
        )
    print("=======================")


if __name__ == "__main__":
    main()
