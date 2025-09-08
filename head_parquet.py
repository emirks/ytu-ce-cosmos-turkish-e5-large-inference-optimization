import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import json
import sys

# Minimal parameters
PARQUET_PATH = "merged_unique_documents.parquet"
NUM_ROWS = 100000
JSONL_OUTPUT_PATH = "head_content.jsonl"


def main():
    pf = pq.ParquetFile(PARQUET_PATH)

    if "mdtext" not in pf.schema.names:
        print("Column 'mdtext' not found; nothing to write.")
        return

    # Read only the first NUM_ROWS for 'mdtext'
    batch_iter = pf.iter_batches(batch_size=NUM_ROWS, columns=["mdtext"])
    try:
        first_batch = next(batch_iter)
    except StopIteration:
        print("File is empty.")
        return

    table = pa.Table.from_batches([first_batch])
    df = table.to_pandas(types_mapper=pd.ArrowDtype)

    values = df["mdtext"].tolist()
    total = len(values)

    # Prepare a tqdm-like iterator with performant fallback
    try:
        from tqdm import tqdm  # type: ignore

        iterator = tqdm(
            values,
            total=total,
            unit="row",
            mininterval=0.2,
            dynamic_ncols=True,
            leave=False,
        )
        using_tqdm = True
    except Exception:
        using_tqdm = False

        def iter_with_progress(it, total_count):
            # Update at most ~50 times to keep it lightweight
            update_every = max(1, total_count // 50) if total_count else 1
            for idx, item in enumerate(it, 1):
                if idx == 1 or idx % update_every == 0 or idx == total_count:
                    pct = int(idx * 100 / total_count) if total_count else 100
                    sys.stdout.write(f"\rWriting JSONL: {idx}/{total_count} ({pct}%)")
                    sys.stdout.flush()
                yield item
            sys.stdout.write("\n")
            sys.stdout.flush()

        iterator = iter_with_progress(values, total)

    # Save JSONL with only {"content": mdtext}
    with open(JSONL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for v in iterator:
            if v is None or pd.isna(v):
                v = ""
            f.write(json.dumps({"content": v}, ensure_ascii=False) + "\n")

    if "using_tqdm" in locals() and using_tqdm:
        try:
            iterator.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    print(f"Saved content JSONL to {JSONL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
