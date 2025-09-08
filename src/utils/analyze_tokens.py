import json
import math
import os
import sys
from typing import Iterable, List, Tuple

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Transformers is required. Install with: pip install transformers tokenizers"
    ) from exc

# Hardcoded parameters
INPUT_JSONL_PATH = "head_content.jsonl"
MODEL_NAME_OR_PATH = "ytu-ce-cosmos/turkish-e5-large"
BATCH_SIZE = 64
WRITE_COUNTS_PATH = "token_counts.jsonl"  # set to None to disable
STATS_OUTPUT_PATH = "token_stats.json"  # set to None to disable


def try_get_progress(iterator: Iterable, total: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(
            iterator,
            total=total,
            unit="row",
            mininterval=0.2,
            dynamic_ncols=True,
            leave=False,
            desc=desc,
        )
    except Exception:
        # Lightweight fallback: update at most ~50 times
        def iter_with_progress(it, total_count):
            update_every = max(1, total_count // 50) if total_count else 1
            for idx, item in enumerate(it, 1):
                if idx == 1 or idx % update_every == 0 or idx == total_count:
                    pct = int(idx * 100 / total_count) if total_count else 100
                    sys.stdout.write(f"\r{desc}: {idx}/{total_count} ({pct}%)")
                    sys.stdout.flush()
                yield item
            sys.stdout.write("\n")
            sys.stdout.flush()

        return iter_with_progress(iterator, total)


def read_jsonl_in_batches(
    file_path: str, batch_size: int
) -> Iterable[Tuple[List[int], List[str]]]:
    batch_indices: List[int] = []
    batch_texts: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                obj = json.loads(line)
                text = obj.get("content", "")
                if text is None:
                    text = ""
                text = str(text)
            except Exception:
                text = ""
            batch_indices.append(idx)
            batch_texts.append(text)
            if len(batch_texts) >= batch_size:
                yield batch_indices, batch_texts
                batch_indices, batch_texts = [], []
    if batch_texts:
        yield batch_indices, batch_texts


def percentile(sorted_values: List[int], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return float(sorted_values[0])
    if p >= 100:
        return float(sorted_values[-1])
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def analyze_token_counts(
    input_jsonl_path: str,
    model_name_or_path: str,
    batch_size: int,
    write_counts_path: str | None,
    stats_output_path: str | None,
) -> None:
    if not os.path.exists(input_jsonl_path):
        raise SystemExit(f"Input file not found: {input_jsonl_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    token_counts: List[int] = []
    total_rows = sum(1 for _ in open(input_jsonl_path, "r", encoding="utf-8"))

    # Optional per-row counts output
    counts_file = (
        open(write_counts_path, "w", encoding="utf-8") if write_counts_path else None
    )

    batch_iter = read_jsonl_in_batches(input_jsonl_path, batch_size)
    progress_iter = try_get_progress(
        batch_iter, total_rows // batch_size + 1, desc="Tokenizing"
    )

    for idx_list, texts in progress_iter:
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # encoded["input_ids"] is a list of lists
        lengths = [len(ids) for ids in encoded["input_ids"]]
        token_counts.extend(lengths)
        if counts_file is not None:
            for i, length in zip(idx_list, lengths):
                counts_file.write(
                    json.dumps({"index": i, "tokens": int(length)}, ensure_ascii=False)
                    + "\n"
                )

    if counts_file is not None:
        counts_file.close()

    token_counts.sort()
    n = len(token_counts)
    min_tokens = token_counts[0] if n else 0
    max_tokens = token_counts[-1] if n else 0
    mean_tokens = (sum(token_counts) / n) if n else 0.0

    p50 = percentile(token_counts, 50)
    p90 = percentile(token_counts, 90)
    p95 = percentile(token_counts, 95)
    p99 = percentile(token_counts, 99)

    stats = {
        "model": model_name_or_path,
        "input_file": input_jsonl_path,
        "count": n,
        "min": int(min_tokens),
        "max": int(max_tokens),
        "mean": mean_tokens,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
    }

    print("\nToken count statistics:")
    for k in ["count", "min", "max", "mean", "p50", "p90", "p95", "p99"]:
        print(f"- {k}: {stats[k]}")

    if stats_output_path:
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Saved stats to {stats_output_path}")


def main():
    analyze_token_counts(
        input_jsonl_path=INPUT_JSONL_PATH,
        model_name_or_path=MODEL_NAME_OR_PATH,
        batch_size=BATCH_SIZE,
        write_counts_path=WRITE_COUNTS_PATH,
        stats_output_path=STATS_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
