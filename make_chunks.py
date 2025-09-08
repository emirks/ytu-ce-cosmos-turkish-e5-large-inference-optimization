import hashlib
import os
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer  # type: ignore
from concurrent.futures import ProcessPoolExecutor

# Config
INPUT_PARQUET = "merged_unique_documents.parquet"
OUTPUT_PARQUET = "chunks.parquet"
MODEL_ID = "ytu-ce-cosmos/turkish-e5-large"
TEXT_COL_IN = "mdtext"
DOC_ID_COL_IN = "documentid"
CHUNK_TOKENS = 512
CHUNK_OVERLAP = 64
MILLION_INDEX = 1  # which million to process (e.g., 0 => 0m-1m, 1 => 1m-2m)
START_DOC = MILLION_INDEX * 1_000_000  # starting document index (0-based)
NUM_DOCS = 10  # number of documents to process from START_DOC
NUM_WORKERS = 4  # increase to CPU cores if desired
FLUSH_EVERY_CHUNKS = 50000
PREFIX = "passage: "  # informational; not applied here, embedding script adds prefix

# Lazy-initialized per-process tokenizer
tokenizer_singleton = None


def get_tokenizer():
    global tokenizer_singleton
    if tokenizer_singleton is None:
        tokenizer_singleton = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    return tokenizer_singleton


def create_chunks_for_text(
    text: str, chunk_tokens: int, overlap: int, tokenizer
) -> List[Tuple[int, int, int]]:
    # Returns list of (start_char, end_char, num_tokens) for each chunk
    if not text:
        return []
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=False,
    )
    offsets = enc["offset_mapping"]
    num_tokens_total = len(offsets)
    if num_tokens_total == 0:
        return []

    stride = max(1, chunk_tokens - overlap)
    chunks: List[Tuple[int, int, int]] = []
    for start_tok in range(0, num_tokens_total, stride):
        end_tok = min(start_tok + chunk_tokens, num_tokens_total)
        start_char = offsets[start_tok][0]
        end_char = offsets[end_tok - 1][1]
        if end_char <= start_char:
            continue
        chunks.append((start_char, end_char, end_tok - start_tok))
        if end_tok >= num_tokens_total:
            break
    return chunks


def process_document(doc_id_raw, text) -> Dict[str, List]:
    # Runs in worker process
    if text is None:
        return {
            "id": [],
            "doc_id": [],
            "chunk_id": [],
            "start_char": [],
            "end_char": [],
            "mdtext": [],
            "num_tokens": [],
            "source_path": [],
            "doc_checksum": [],
        }
    if not isinstance(text, str):
        text = str(text)
    doc_id = str(doc_id_raw)

    tok = get_tokenizer()
    chunk_spans = create_chunks_for_text(text, CHUNK_TOKENS, CHUNK_OVERLAP, tok)
    if not chunk_spans:
        return {
            "id": [],
            "doc_id": [],
            "chunk_id": [],
            "start_char": [],
            "end_char": [],
            "mdtext": [],
            "num_tokens": [],
            "source_path": [],
            "doc_checksum": [],
        }

    h_doc = hashlib.sha1(text.encode("utf-8")).hexdigest()

    out: Dict[str, List] = {
        "id": [],
        "doc_id": [],
        "chunk_id": [],
        "start_char": [],
        "end_char": [],
        "mdtext": [],
        "num_tokens": [],
        "source_path": [],
        "doc_checksum": [],
    }
    for cid, (s, e, ntok) in enumerate(chunk_spans):
        chunk_text = text[s:e]
        h = hashlib.sha1()
        h.update(doc_id.encode("utf-8"))
        h.update(b":")
        h.update(str(cid).encode("utf-8"))
        h.update(b"@")
        h.update(str(s).encode("utf-8"))
        h.update(b"-")
        h.update(str(e).encode("utf-8"))
        rid = h.hexdigest()

        out["id"].append(rid)
        out["doc_id"].append(doc_id)
        out["chunk_id"].append(int(cid))
        out["start_char"].append(int(s))
        out["end_char"].append(int(e))
        out["mdtext"].append(chunk_text)
        out["num_tokens"].append(int(ntok))
        out["source_path"].append(INPUT_PARQUET)
        out["doc_checksum"].append(h_doc)

    return out


def format_million_label(index_value: int) -> str:
    millions_float = index_value / 1_000_000
    s = f"{millions_float:.1f}".rstrip("0").rstrip(".")
    return f"{s}m"


def main():
    pf = pq.ParquetFile(INPUT_PARQUET)

    # Define output schema and writer
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("doc_id", pa.string()),
            ("chunk_id", pa.int32()),
            ("start_char", pa.int32()),
            ("end_char", pa.int32()),
            ("mdtext", pa.string()),
            ("num_tokens", pa.int32()),
            ("source_path", pa.string()),
            ("doc_checksum", pa.string()),
        ]
    )
    start_label = format_million_label(START_DOC)
    target_end_label = format_million_label(START_DOC + NUM_DOCS)
    output_parquet = f"chunks_{start_label}_{target_end_label}.parquet"
    writer = pq.ParquetWriter(output_parquet, schema)

    # Buffer for incremental flush
    buf: Dict[str, List] = {k: [] for k in schema.names}

    def flush_if_needed(force: bool = False):
        if not buf["id"]:
            return
        if (len(buf["id"]) >= FLUSH_EVERY_CHUNKS) or force:
            table = pa.Table.from_pydict(buf, schema=schema)
            writer.write_table(table)
            # clear buffers
            for k in buf:
                buf[k].clear()

    docs_skipped = 0
    docs_processed = 0
    batch_size_rows = 2048
    columns = [DOC_ID_COL_IN, TEXT_COL_IN]

    pbar = tqdm(
        total=NUM_DOCS,
        desc=f"Chunking docs (parallel) [{start_label}..{target_end_label})",
    )
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for batch in pf.iter_batches(batch_size=batch_size_rows, columns=columns):
            doc_col = batch.column(0).to_pylist()
            text_col = batch.column(1).to_pylist()

            if docs_processed >= NUM_DOCS:
                break

            if docs_skipped < START_DOC:
                to_skip = min(len(doc_col), START_DOC - docs_skipped)
                docs_skipped += to_skip
                if to_skip >= len(doc_col):
                    continue
                doc_col = doc_col[to_skip:]
                text_col = text_col[to_skip:]

            remaining = max(0, NUM_DOCS - docs_processed)
            if remaining <= 0:
                break
            slice_end = min(len(doc_col), remaining)

            # Map over two iterables (Windows-safe; no lambdas)
            for res in ex.map(
                process_document,
                doc_col[:slice_end],
                text_col[:slice_end],
                chunksize=16,
            ):
                if res["id"]:
                    for k in schema.names:
                        buf[k].extend(res[k])
                pbar.update(1)
                docs_processed += 1
                flush_if_needed()

    pbar.close()

    # Final flush and close
    flush_if_needed(force=True)
    writer.close()

    actual_end_label = format_million_label(START_DOC + docs_processed)
    if actual_end_label != target_end_label:
        final_output_parquet = f"chunks_{start_label}_{actual_end_label}.parquet"
        try:
            os.replace(output_parquet, final_output_parquet)
            output_parquet = final_output_parquet
        except Exception:
            pass

    print(
        f"Wrote to {output_parquet} (docs processed: {docs_processed} from {START_DOC} to {START_DOC + docs_processed})"
    )


if __name__ == "__main__":
    main()
