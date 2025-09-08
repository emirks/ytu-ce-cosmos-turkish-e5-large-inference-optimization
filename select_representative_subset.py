import argparse
import os
import random
from typing import List, Optional, Tuple, Dict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Optional: fallback to sklearn if faiss isn't available
try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def embedding_batch_to_numpy(arr: pa.Array) -> np.ndarray:
    # Handles ListArray and FixedSizeListArray
    typ = arr.type
    if pa.types.is_fixed_size_list(typ):
        width = typ.list_size
        values = arr.values().to_numpy(zero_copy_only=False)
        out = values.reshape(-1, width).astype("float32", copy=False)
        return out
    if pa.types.is_list(typ):
        # Generic path; a bit slower but robust
        py = arr.to_pylist()
        out = np.asarray(py, dtype="float32")
        return out
    raise ValueError("Embedding column must be a list or fixed_size_list Arrow type.")


def reservoir_like_subsample(
    total_rows: int, desired: int, batch_lens: List[int], rng: np.random.Generator
) -> List[np.ndarray]:
    # Returns per-batch boolean masks selecting approximately 'desired' rows overall
    p = desired / max(total_rows, 1)
    remain = desired
    remain_rows = total_rows
    masks = []
    for n in batch_lens:
        # Target expected selection for this batch
        target = int(round(p * n)) if remain_rows > 0 else 0
        target = min(target, remain)
        if target <= 0:
            masks.append(np.zeros(n, dtype=bool))
        else:
            idx = np.arange(n)
            rng.shuffle(idx)
            sel = np.zeros(n, dtype=bool)
            sel[idx[:target]] = True
            masks.append(sel)
        remain -= masks[-1].sum()
        remain_rows -= n
        if remain_rows > 0:
            p = remain / remain_rows if remain > 0 else 0.0
    # If due to rounding we missed a bit, distribute over earlier batches
    if remain > 0:
        for i, n in enumerate(batch_lens):
            if remain <= 0:
                break
            free = (~masks[i]).sum()
            add = min(free, remain)
            if add > 0:
                idx = np.where(~masks[i])[0]
                rng.shuffle(idx)
                masks[i][idx[:add]] = True
                remain -= add
    return masks


def train_kmeans_faiss(
    dim: int,
    k: int,
    train_vectors: np.ndarray,
    spherical: bool,
    niter: int = 25,
    seed: int = 123,
) -> np.ndarray:
    if spherical:
        train_vectors = l2_normalize_rows(train_vectors)
    kmeans = faiss.Kmeans(
        d=dim, k=k, niter=niter, verbose=True, seed=seed, spherical=spherical
    )
    kmeans.train(train_vectors)
    return faiss.vector_float_to_array(kmeans.centroids).reshape(k, dim)


def train_kmeans_sklearn(
    dim: int,
    k: int,
    train_vectors: np.ndarray,
    spherical: bool,
    niter: int = 50,
    seed: int = 123,
) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    if spherical:
        train_vectors = l2_normalize_rows(train_vectors)
    mbk = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        batch_size=8192,
        n_init=1,
        max_iter=niter,
        random_state=seed,
        verbose=1,
    )
    mbk.fit(train_vectors)
    return mbk.cluster_centers_.astype("float32", copy=False)


def build_centroid_index(centroids: np.ndarray, use_l2: bool = True):
    d = centroids.shape[1]
    if use_l2:
        index = faiss.IndexFlatL2(d)
    else:
        index = faiss.IndexFlatIP(d)
    index.add(centroids)
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Select a representative 100k subset via clustering medoids."
    )
    parser.add_argument("--input", required=True, help="Path to input Parquet file")
    parser.add_argument(
        "--output", required=True, help="Path to output Parquet file with selected rows"
    )
    parser.add_argument(
        "--embedding-col", default="embedding", help="Embedding column name"
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="ID column name (optional). If omitted, uses global row index.",
    )
    parser.add_argument(
        "--k", type=int, default=100_000, help="Number of representatives to select"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=1_000_000,
        help="Training subsample size for k-means",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50_000, help="Rows per streaming batch"
    )
    parser.add_argument(
        "--spherical",
        action="store_true",
        help="Use cosine (L2-normalize) instead of raw L2",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--keep-cols",
        nargs="*",
        default=None,
        help="Columns to keep in the output (default: all columns)",
    )
    parser.add_argument(
        "--use-sklearn",
        action="store_true",
        help="Force sklearn MiniBatchKMeans instead of FAISS",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    pf = pq.ParquetFile(args.input)
    total_rows = pf.metadata.num_rows
    if args.k > total_rows:
        raise ValueError(f"k={args.k} exceeds dataset size {total_rows}")

    # Discover schema and dimension
    first_batch = next(
        pf.iter_batches(
            batch_size=min(args.batch_size, 10000), columns=[args.embedding_col]
        )
    )
    first_embs = embedding_batch_to_numpy(
        first_batch.column(first_batch.schema.get_field_index(args.embedding_col))
    )
    dim = first_embs.shape[1]

    # Decide columns to read/keep
    if args.keep_cols is None:
        keep_cols = [name for name in pf.schema.names]  # keep all
    else:
        keep_cols = args.keep_cols
        if args.embedding_col not in keep_cols:
            keep_cols = [args.embedding_col] + keep_cols
        if args.id_col and args.id_col not in keep_cols:
            keep_cols = [args.id_col] + keep_cols

    # Pass 0: compute per-batch sizes for subsampling masks
    batch_sizes = []
    for rb in pf.iter_batches(batch_size=args.batch_size, columns=[args.embedding_col]):
        batch_sizes.append(len(rb))

    # Build masks for training subset selection
    train_masks = reservoir_like_subsample(
        total_rows, min(args.train_size, total_rows), batch_sizes, rng
    )

    # Pass 1: collect training vectors
    train_vecs = []
    idx_batch = 0
    for rb in tqdm(
        pf.iter_batches(batch_size=args.batch_size, columns=[args.embedding_col]),
        desc="Collecting train vectors",
    ):
        mask = train_masks[idx_batch]
        if mask.any():
            embs = embedding_batch_to_numpy(
                rb.column(rb.schema.get_field_index(args.embedding_col))
            )
            train_vecs.append(embs[mask])
        idx_batch += 1
    train_vecs = np.concatenate(train_vecs, axis=0).astype("float32", copy=False)

    # Train k-means
    if args.use_sklearn or not HAS_FAISS:
        centroids = train_kmeans_sklearn(
            dim, args.k, train_vecs, spherical=args.spherical, seed=args.seed
        )
        use_l2 = not args.spherical
        index = None
    else:
        centroids = train_kmeans_faiss(
            dim, args.k, train_vecs, spherical=args.spherical, seed=args.seed
        )
        use_l2 = True if not args.spherical else False  # cosine -> IP
        index = build_centroid_index(centroids, use_l2=use_l2)

    # Pass 2: assign all points to nearest centroid and keep closest (medoid)
    best_dist = np.full(args.k, np.inf, dtype="float32")
    best_ids = np.full(args.k, -1, dtype=np.int64)
    use_row_index_as_id = args.id_col is None
    global_row_index = 0

    # If no FAISS (sklearn path), build a search structure over centroids using numpy
    if index is None:
        # Pre-normalize centroids if IP
        centroids_search = centroids.copy()
        if not use_l2:
            centroids_search = l2_normalize_rows(centroids_search)

    for rb in tqdm(
        pf.iter_batches(
            batch_size=args.batch_size,
            columns=(
                [args.embedding_col, args.id_col]
                if args.id_col
                else [args.embedding_col]
            ),
        ),
        desc="Selecting representatives",
    ):
        embs = embedding_batch_to_numpy(
            rb.column(rb.schema.get_field_index(args.embedding_col))
        )
        if args.spherical:
            embs = l2_normalize_rows(embs)

        if use_row_index_as_id:
            ids = np.arange(
                global_row_index, global_row_index + len(embs), dtype=np.int64
            )
        else:
            id_arr = rb.column(rb.schema.get_field_index(args.id_col))
            ids = np.asarray(id_arr.to_numpy(zero_copy_only=False))

        if index is not None:
            k = 1
            D, I = index.search(embs.astype("float32", copy=False), k)
            dists = D[:, 0]
            assigns = I[:, 0]
            if not use_l2:
                # For IP, smaller distance is larger IP; convert to "distance"
                dists = -dists
        else:
            # Numpy brute-force over centroids
            if use_l2:
                # (a-b)^2 = a^2 + b^2 - 2ab; but we can do direct
                # Compute distances to centroids in chunks to avoid RAM spikes
                chunk = 8192
                assigns = np.empty(len(embs), dtype=np.int32)
                dists = np.empty(len(embs), dtype=np.float32)
                for start in range(0, len(embs), chunk):
                    end = start + chunk
                    e = embs[start:end]
                    # Broadcasting to compute squared L2 distances
                    # dist^2 = sum((e - c)^2)
                    # Use efficient computation
                    e2 = (e * e).sum(axis=1, keepdims=True)
                    c2 = (centroids**2).sum(axis=1)
                    cross = e @ centroids.T
                    dist2 = e2 + c2[None, :] - 2 * cross
                    a = np.argmin(dist2, axis=1)
                    d = dist2[np.arange(len(a)), a]
                    assigns[start:end] = a.astype(np.int32)
                    dists[start:end] = d.astype(np.float32)
            else:
                # Cosine/IP: pick max dot
                chunk = 8192
                assigns = np.empty(len(embs), dtype=np.int32)
                dists = np.empty(len(embs), dtype=np.float32)
                for start in range(0, len(embs), chunk):
                    end = start + chunk
                    e = embs[start:end]
                    sims = e @ centroids_search.T
                    a = np.argmax(sims, axis=1)
                    s = sims[np.arange(len(a)), a]
                    assigns[start:end] = a.astype(np.int32)
                    dists[start:end] = (-s).astype(
                        np.float32
                    )  # convert to "distance" as negative similarity

        # Update best per centroid
        for cid, dist, rid in zip(assigns, dists, ids):
            if dist < best_dist[cid]:
                best_dist[cid] = dist
                best_ids[cid] = rid

        global_row_index += len(embs)

    # Some centroids might be empty (rare). Filter them out.
    chosen_ids = best_ids[best_ids >= 0]
    if len(chosen_ids) < args.k:
        print(
            f"Warning: only {len(chosen_ids)} centroids received points. Output will have {len(chosen_ids)} rows."
        )
    chosen_set = set(int(x) for x in chosen_ids.tolist())

    # Pass 3: write selected rows to output
    writer = None
    written = 0
    cols_to_read = keep_cols
    for rb in tqdm(
        pf.iter_batches(batch_size=args.batch_size, columns=cols_to_read),
        desc="Writing output",
    ):
        if args.id_col:
            id_arr = rb.column(rb.schema.get_field_index(args.id_col)).to_numpy(
                zero_copy_only=False
            )
            mask = np.array([int(x) in chosen_set for x in id_arr], dtype=bool)
        else:
            # Use global row indices
            start = global_row_index
            # Note: We advanced global_row_index earlier; reset for this pass
            pass

        # For global indices, recompute per batch
        # We'll keep a separate counter
        # To avoid confusion, we compute batch start on the fly
        # Re-scan with a running counter
        # Implement separate loop to handle both paths cleanly
    # Re-implement pass 3 with a separate row counter
    writer = None
    written = 0
    row_counter = 0
    for rb in pf.iter_batches(batch_size=args.batch_size, columns=cols_to_read):
        n = len(rb)
        if args.id_col:
            id_arr = rb.column(rb.schema.get_field_index(args.id_col)).to_numpy(
                zero_copy_only=False
            )
            mask = np.array([int(x) in chosen_set for x in id_arr], dtype=bool)
        else:
            ids_here = np.arange(row_counter, row_counter + n, dtype=np.int64)
            mask = np.isin(ids_here, np.fromiter(chosen_set, dtype=np.int64))
        if mask.any():
            table = pa.Table.from_batches([rb]).filter(pa.array(mask))
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
            writer.write_table(table)
            written += table.num_rows
        row_counter += n
    if writer is not None:
        writer.close()

    print(f"Done. Wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
