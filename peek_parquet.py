import pyarrow as pa
import pyarrow.parquet as pq

aPATH = "chunks_1m_1m.parquet"
NUM_ROWS = 10
TRUNCATE = 50


def truncate_value(v, limit):
    if v is None:
        return None
    s = v if isinstance(v, str) else str(v)
    if limit > 0 and len(s) > limit:
        return s[: limit - 3] + "..."
    return s


def main():
    pf = pq.ParquetFile(aPATH)
    print(f"rows= {pf.metadata.num_rows}  cols= {pf.metadata.num_columns}")
    print("columns=", pf.schema.names)
    print("\nSchema:\n", pf.schema)

    try:
        first_batch = next(pf.iter_batches(batch_size=NUM_ROWS))
    except StopIteration:
        print("\n(empty file)")
        return

    table = pa.Table.from_batches([first_batch])
    cols = table.column_names
    data = {c: table[c].to_pylist() for c in cols}
    n = min(NUM_ROWS, table.num_rows)

    print(f"\nFirst {n} rows (truncated to {TRUNCATE} chars):")
    for i in range(n):
        row = {}
        for c in cols:
            v = data[c][i]
            if isinstance(v, str):
                row[c] = truncate_value(v, TRUNCATE)
            else:
                row[c] = v
        print(row)


if __name__ == "__main__":
    main()
