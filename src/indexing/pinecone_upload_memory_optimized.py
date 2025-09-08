#!/usr/bin/env python3
"""
MEMORY-OPTIMIZED Pinecone Upload Script for Turkish Legal Documents
Processes batches on-demand to minimize RAM usage
"""

import os
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple
from tqdm import tqdm
import time
import re
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC  # For better performance


# Configuration
class Config:
    # Pinecone settings - Hardcoded values
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "ytu-ce-cosmos-turkish-e5-large-embeddings"
    PINECONE_ENVIRONMENT = "europe-west4"
    PINECONE_CLOUD = "gcp"

    BASE_DIR = Path("/content/embeddings/docs_0m_1m")
    MAIN_METADATA_PATH = (
        "/content/merged_unique_documents.parquet"  # Main metadata file
    )

    # File paths (will be auto-detected or can be set manually)
    EMBEDDINGS_PATH = None  # Path to .fp16.npy file
    PARQUET_PATH = None  # Path to parquet file
    META_PATH = None  # Path to meta.json file

    # Upload settings - OPTIMIZED for speed AND memory
    BATCH_SIZE = 500  # Larger batches for better throughput
    MAX_WORKERS = 20  # Concurrent async uploads
    NAMESPACE = ""  # Default namespace (empty string for default)
    POOL_THREADS = 50  # High thread count for async requests

    # Embedding settings
    EMBEDDING_DIMENSION = 1024  # Turkish E5 model dimension
    DISTANCE_METRIC = "cosine"  # cosine, euclidean, or dotproduct


class MemoryOptimizedPineconeUploader:
    def __init__(self, config: Config):
        self.config = config
        self.pc = None
        self.index = None
        self.embeddings = None  # Memory-mapped, not loaded entirely
        self.missing_docs = []

        # For streaming processing
        self.chunk_columns = [
            "id",
            "doc_id",
            "chunk_id",
            "start_char",
            "end_char",
            "mdtext",
            "num_tokens",
            "source_path",
            "doc_checksum",
        ]
        self.main_columns = [
            "documentid",
            "itemtype_name",
            "itemtype_description",
            "birimadi",
            "esasnoyil",
            "esasnosira",
            "kararnoyil",
            "kararnosira",
            "karartarihi",
            "karartarihistr",
            "kesinlesmedurumu",
            "kararno",
            "esasno",
        ]

    def load_files_memory_efficient(
        self,
        embeddings_path: str = None,
        parquet_path: str = None,
        meta_path: str = None,
    ):
        """Load files with minimal memory footprint"""

        # Auto-detect file paths if not provided
        if not embeddings_path:
            embeddings_path = self._auto_detect_files()[0]
        if not parquet_path:
            parquet_path = self._auto_detect_files()[1]
        if not meta_path:
            meta_path = self._auto_detect_files()[2]

        print(f"Loading files:")
        print(f"  Embeddings: {embeddings_path}")
        print(f"  Parquet: {parquet_path}")
        print(f"  Metadata: {meta_path}")

        # Load embeddings as memory-mapped (efficient)
        self.embeddings = np.memmap(embeddings_path, dtype=np.float16, mode="r")

        # Load metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta_info = json.load(f)

        # Reshape embeddings based on metadata
        n_vectors = self.meta_info["n_vectors"]
        dim = self.meta_info["dim"]
        self.embeddings = self.embeddings.reshape(n_vectors, dim)

        # Store paths for streaming processing
        self.parquet_path = parquet_path
        self.main_metadata_path = self.config.MAIN_METADATA_PATH

        print(f"✓ Memory-mapped {n_vectors} embeddings with dimension {dim}")
        print(f"✓ Set up for streaming parquet processing")

    def _auto_detect_files(self) -> Tuple[str, str, str]:
        """Auto-detect embedding, parquet, and meta files"""
        base_dir = Path("/content/embeddings")

        embeddings_files = list(base_dir.glob("**/embeddings_*.fp16.npy"))
        parquet_files = list(base_dir.glob("**/chunks_*.parquet"))
        meta_files = list(base_dir.glob("**/meta_*.json"))

        if not embeddings_files or not parquet_files or not meta_files:
            raise FileNotFoundError(
                f"Could not auto-detect files. Found: "
                f"{len(embeddings_files)} embedding files, "
                f"{len(parquet_files)} parquet files, "
                f"{len(meta_files)} meta files"
            )

        return str(embeddings_files[0]), str(parquet_files[0]), str(meta_files[0])

    def initialize_pinecone(self):
        """Initialize Pinecone connection"""
        if not self.config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        # Use gRPC client with optimized settings
        self.pc = PineconeGRPC(
            api_key=self.config.PINECONE_API_KEY, pool_threads=self.config.POOL_THREADS
        )

        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.config.PINECONE_INDEX_NAME not in existing_indexes:
            print(f"Creating new index: {self.config.PINECONE_INDEX_NAME}")
            self.pc.create_index(
                name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.EMBEDDING_DIMENSION,
                metric=self.config.DISTANCE_METRIC,
                spec=ServerlessSpec(
                    cloud=self.config.PINECONE_CLOUD,
                    region=self.config.PINECONE_ENVIRONMENT,
                ),
            )

            # Wait for index to be ready
            while not self.pc.describe_index(self.config.PINECONE_INDEX_NAME).status[
                "ready"
            ]:
                print("Waiting for index to be ready...")
                time.sleep(5)
        else:
            print(f"Using existing index: {self.config.PINECONE_INDEX_NAME}")

        self.index = self.pc.Index(
            self.config.PINECONE_INDEX_NAME, pool_threads=self.config.POOL_THREADS
        )

    def stream_merged_data(self) -> Generator[Tuple[int, Dict], None, None]:
        """🚀 FAST streaming with pandas merge (memory-efficient chunked processing)"""

        print("🔄 Starting fast pandas streaming merge...")

        # Load main metadata once (smaller file)
        print(f"📊 Loading main metadata from: {self.main_metadata_path}")
        main_df = pd.read_parquet(
            self.main_metadata_path, columns=self.main_columns, use_threads=True
        )
        main_df["documentid"] = main_df["documentid"].astype(str)

        # Helper functions for vectorized processing
        def safe_value_series(series, default=""):
            """Vectorized version of safe_value for pandas Series"""
            return series.fillna(default).astype(str).where(series.notna(), default)

        def safe_int_series(series, default=0):
            """Safely convert series to int, replacing nulls with default"""
            return pd.to_numeric(series, errors="coerce").fillna(default).astype(int)

        print(f"✅ Loaded main metadata: {len(main_df):,} documents")

        # Stream chunk data in smaller batches and merge with pandas (FAST!)
        print("📊 Streaming chunk data with fast pandas merge...")
        chunk_size = 100000  # Process 100K rows at a time (larger for efficiency)

        parquet_file = pq.ParquetFile(self.parquet_path)
        total_rows = parquet_file.metadata.num_rows
        processed_rows = 0

        for batch_start in range(0, total_rows, chunk_size):
            batch_end = min(batch_start + chunk_size, total_rows)

            # Read chunk batch
            chunk_df = pd.read_parquet(
                self.parquet_path, columns=self.chunk_columns, use_threads=True
            ).iloc[batch_start:batch_end]

            chunk_df["doc_id"] = chunk_df["doc_id"].astype(str)

            # 🚀 FAST PANDAS MERGE (vectorized operation)
            merged_batch = chunk_df.merge(
                main_df,
                left_on="doc_id",
                right_on="documentid",
                how="left",
                suffixes=("_chunk", "_main"),
            )

            # 🚀 FAST vectorized processing (like the original optimized version)
            merged_batch["documentid"] = merged_batch["documentid"].fillna(
                merged_batch["doc_id"]
            )
            merged_batch["itemtype_name"] = merged_batch["itemtype_name"].fillna(
                "UNKNOWN"
            )
            merged_batch["itemtype_description"] = merged_batch[
                "itemtype_description"
            ].fillna("Unknown Document Type")
            merged_batch["birimadi"] = merged_batch["birimadi"].fillna("Unknown")

            # Vectorized processing of all fields
            merged_batch["proc_chunk_id"] = safe_value_series(merged_batch["chunk_id"])
            merged_batch["proc_doc_checksum"] = safe_value_series(
                merged_batch["doc_checksum"]
            )
            merged_batch["proc_mdtext"] = safe_value_series(merged_batch["mdtext"])
            merged_batch["proc_documentid"] = safe_value_series(
                merged_batch["documentid"]
            )
            merged_batch["proc_itemtype_name"] = safe_value_series(
                merged_batch["itemtype_name"]
            )
            merged_batch["proc_itemtype_description"] = safe_value_series(
                merged_batch["itemtype_description"]
            )
            merged_batch["proc_birimadi"] = safe_value_series(merged_batch["birimadi"])
            merged_batch["proc_karartarihi"] = safe_value_series(
                merged_batch["karartarihi"]
            )
            merged_batch["proc_karartarihistr"] = safe_value_series(
                merged_batch["karartarihistr"]
            )
            merged_batch["proc_kesinlesmedurumu"] = safe_value_series(
                merged_batch["kesinlesmedurumu"]
            )
            merged_batch["proc_kararno"] = safe_value_series(merged_batch["kararno"])
            merged_batch["proc_esasno"] = safe_value_series(merged_batch["esasno"])

            # Handle integer fields
            merged_batch["proc_start_char"] = safe_int_series(
                merged_batch["start_char"]
            )
            merged_batch["proc_end_char"] = safe_int_series(merged_batch["end_char"])
            merged_batch["proc_num_tokens"] = safe_int_series(
                merged_batch["num_tokens"]
            )
            merged_batch["proc_esasnoyil"] = safe_int_series(merged_batch["esasnoyil"])
            merged_batch["proc_esasnosira"] = safe_int_series(
                merged_batch["esasnosira"]
            )
            merged_batch["proc_kararnoyil"] = safe_int_series(
                merged_batch["kararnoyil"]
            )
            merged_batch["proc_kararnosira"] = safe_int_series(
                merged_batch["kararnosira"]
            )

            # Track missing documents
            missing_mask = merged_batch["proc_itemtype_name"] == "UNKNOWN"
            if missing_mask.any():
                missing_in_batch = (
                    merged_batch[missing_mask]["doc_id"].unique().tolist()
                )
                self.missing_docs.extend(missing_in_batch)

            # Yield each row with processed data (fast iteration over processed batch)
            for idx, row in merged_batch.iterrows():
                unified_metadata = {
                    # From chunk parquet (essential fields only)
                    "documentId": row["proc_documentid"],
                    "chunkId": row["proc_chunk_id"],
                    "start_char": row["proc_start_char"],
                    "end_char": row["proc_end_char"],
                    "num_tokens": row["proc_num_tokens"],
                    "doc_checksum": row["proc_doc_checksum"],
                    # From main parquet (essential fields only)
                    "itemTypeName": row["proc_itemtype_name"],
                    "itemTypeDescription": row["proc_itemtype_description"],
                    "birimAdi": row["proc_birimadi"],
                    "esasNoYil": row["proc_esasnoyil"],
                    "esasNoSira": row["proc_esasnosira"],
                    "kararNoYil": row["proc_kararnoyil"],
                    "kararNoSira": row["proc_kararnosira"],
                    "kararTarihi": row["proc_karartarihi"],
                    "kararTarihiStr": row["proc_karartarihistr"],
                    "kesinlesmeDurumu": row["proc_kesinlesmedurumu"],
                    "kararNo": row["proc_kararno"],
                    "esasNo": row["proc_esasno"],
                    "mdtext": row["proc_mdtext"],
                }

                # Use the original index from the chunk (batch_start + relative position)
                original_idx = batch_start + (idx - merged_batch.index[0])
                yield original_idx, unified_metadata

            processed_rows += len(merged_batch)
            print(
                f"   ✅ Processed {processed_rows:,}/{total_rows:,} rows ({processed_rows/total_rows*100:.1f}%)"
            )

            del chunk_df, merged_batch  # Free memory after each batch

        del main_df  # Free main metadata memory

    def prepare_vectors_for_upload_streaming(self) -> Generator[List[Dict], None, None]:
        """Memory-efficient batch preparation using streaming"""

        batch = []
        total_processed = 0

        for idx, metadata in self.stream_merged_data():
            # Get embedding vector (convert from float16 to float32 for Pinecone)
            vector_values = self.embeddings[idx].astype(np.float32).tolist()

            # Create vector record
            vector_record = {
                "id": f"doc_{metadata['documentId']}_chunk_{metadata['chunkId']}",
                "values": vector_values,
                "metadata": metadata,
            }

            batch.append(vector_record)
            total_processed += 1

            # Yield batch when it reaches desired size
            if len(batch) >= self.config.BATCH_SIZE:
                yield batch
                batch = []

                # Progress update
                if total_processed % 10000 == 0:
                    print(f"📝 Prepared {total_processed:,} vectors for upload")

        # Yield remaining vectors
        if batch:
            yield batch

        print(f"✅ Total prepared: {total_processed:,} vectors")

    def upload_to_pinecone_memory_efficient(self):
        """Memory-efficient upload using streaming"""
        print(
            f"🚀 Starting memory-optimized upload to: {self.config.PINECONE_INDEX_NAME}"
        )
        print(
            f"⚙️  Using {self.config.BATCH_SIZE} vectors/batch, {self.config.POOL_THREADS} threads"
        )

        total_uploaded = 0
        failed_batches = []
        upload_start_time = time.time()
        chunk_async_results = []
        batch_count = 0

        # Stream batches and upload
        for batch in self.prepare_vectors_for_upload_streaming():
            try:
                # Upload batch asynchronously
                async_result = self.index.upsert(
                    vectors=batch,
                    namespace=self.config.NAMESPACE,
                    async_req=True,  # ASYNC upload for speed
                )
                chunk_async_results.append((batch_count, async_result, len(batch)))
                batch_count += 1

                # Process in chunks to avoid overwhelming API
                if len(chunk_async_results) >= self.config.MAX_WORKERS:
                    print(
                        f"🔄 Processing async chunk ({len(chunk_async_results)} batches)"
                    )

                    # Wait for this chunk to complete
                    for batch_idx, async_result, batch_size in chunk_async_results:
                        try:
                            async_result.get()  # Wait for completion
                            total_uploaded += batch_size

                            if batch_idx % 10 == 0:  # Progress update every 10 batches
                                elapsed_time = time.time() - upload_start_time
                                upload_rate = (
                                    total_uploaded / elapsed_time
                                    if elapsed_time > 0
                                    else 0
                                )
                                print(
                                    f"✅ Uploaded {total_uploaded:,} vectors ({batch_idx + 1} batches) - Rate: {upload_rate:.0f} vectors/sec"
                                )

                        except Exception as e:
                            print(f"Failed to complete batch {batch_idx}: {e}")
                            failed_batches.append(batch_idx)

                    # Clear the chunk and add small delay
                    chunk_async_results = []
                    time.sleep(0.1)

            except Exception as e:
                print(f"Failed to submit batch {batch_count}: {e}")
                failed_batches.append(batch_count)

        # Process any remaining batches
        if chunk_async_results:
            print(f"🔄 Processing final chunk ({len(chunk_async_results)} batches)")
            for batch_idx, async_result, batch_size in chunk_async_results:
                try:
                    async_result.get()
                    total_uploaded += batch_size
                except Exception as e:
                    print(f"Failed to complete final batch {batch_idx}: {e}")
                    failed_batches.append(batch_idx)

        print(f"🎉 Upload complete! Total vectors uploaded: {total_uploaded:,}")

        if failed_batches:
            print(f"⚠ Failed batches: {failed_batches}")

        # Report missing documents
        if self.missing_docs:
            unique_missing = list(set(self.missing_docs))
            print(
                f"⚠ {len(unique_missing)} unique documents not found in main metadata"
            )

        # Verify upload
        print("🔍 Verifying upload...")
        time.sleep(5)  # Wait for indexing
        stats = self.index.describe_index_stats()
        print(f"📊 Final index stats: {stats}")

    def run_upload(self):
        """Complete memory-optimized upload pipeline"""
        try:
            print("=== MEMORY-OPTIMIZED Pinecone Upload Pipeline ===")

            # Step 1: Load files efficiently
            self.load_files_memory_efficient()

            # Step 2: Initialize Pinecone
            self.initialize_pinecone()

            # Step 3: Upload vectors with streaming
            self.upload_to_pinecone_memory_efficient()

            print("=== Upload Complete ===")

        except Exception as e:
            print(f"Upload failed: {e}")
            raise


if __name__ == "__main__":
    # Configuration
    config = Config()

    # Create uploader and run
    uploader = MemoryOptimizedPineconeUploader(config)
    uploader.run_upload()
