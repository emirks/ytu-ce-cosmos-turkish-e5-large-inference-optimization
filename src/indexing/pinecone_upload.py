#!/usr/bin/env python3
"""
Pinecone Upload Script for Turkish Legal Documents
Uploads embeddings from numpy memmap and metadata from parquet to Pinecone
Supports both YARGITAYKARARI and YERELHUKUK document types
Processes documents in chunks to prevent RAM crashes
Processes multiple embedding file sets automatically
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
import gc  # For garbage collection
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC  # For better performance


# Configuration
class Config:
    # Pinecone settings - Hardcoded values
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "ytu-ce-cosmos-turkish-e5-large-embeddings"
    PINECONE_ENVIRONMENT = "europe-west4"
    PINECONE_CLOUD = "gcp"

    # Base directory and main metadata path
    BASE_DIR = Path("/content/embeddings")
    MAIN_METADATA_PATH = (
        "/content/merged_unique_documents.parquet"  # Main metadata file
    )

    # File ranges to process - from 1m_2m to 6m_6.6m
    FILE_RANGES = ["1m_2m", "2m_3m", "3m_4m", "4m_5m", "5m_6m", "6m_6.6m"]

    # Processing settings - OPTIMIZED for memory efficiency
    CHUNK_SIZE = 100000  # Process 100k documents at a time to prevent RAM crashes
    BATCH_SIZE = 200  # Larger batches for better throughput (max 1000)
    MAX_WORKERS = 30  # More parallel workers for async uploads
    NAMESPACE = ""  # Default namespace (empty string for default)
    POOL_THREADS = 50  # High thread count for async requests
    CONNECTION_POOL_MAXSIZE = 50  # More cached connections

    # Embedding settings
    EMBEDDING_DIMENSION = 1024  # Turkish E5 model dimension
    DISTANCE_METRIC = "cosine"  # cosine, euclidean, or dotproduct


class PineconeUploader:
    def __init__(self, config: Config):
        self.config = config
        self.pc = None
        self.index = None
        self.embeddings = None
        self.parquet_df = None
        self.meta_info = None
        self.main_metadata_df = None
        self.missing_docs = []  # Track documents not found in main metadata
        self.total_processed = 0  # Track total processed documents across chunks
        self.overall_total_uploaded = 0  # Track total across all file sets
        self.overall_start_time = None

    def find_files_for_range(self, file_range: str) -> Tuple[str, str, str]:
        """Find embedding, parquet, and meta files for a specific range"""

        # Look for files in the specific range directory
        range_dir = self.config.BASE_DIR / f"docs_{file_range}"

        if not range_dir.exists():
            raise FileNotFoundError(f"Directory not found: {range_dir}")

        # Look for patterns like embeddings_1m_2m.fp16.npy, chunks_1m_2m.parquet, meta_1m_2m.json
        embeddings_files = list(range_dir.glob(f"embeddings_{file_range}.fp16.npy"))
        parquet_files = list(range_dir.glob(f"chunks_{file_range}.parquet"))
        meta_files = list(range_dir.glob(f"meta_{file_range}.json"))

        if not embeddings_files or not parquet_files or not meta_files:
            raise FileNotFoundError(
                f"Could not find files for range {file_range}. Found: "
                f"{len(embeddings_files)} embedding files, "
                f"{len(parquet_files)} parquet files, "
                f"{len(meta_files)} meta files in {range_dir}"
            )

        # Use the first matching set
        return str(embeddings_files[0]), str(parquet_files[0]), str(meta_files[0])

    def load_files(
        self,
        embeddings_path: str = None,
        parquet_path: str = None,
        meta_path: str = None,
    ):
        """Load embeddings, parquet data, and metadata"""

        print(f"Loading files:")
        print(f"  Embeddings: {embeddings_path}")
        print(f"  Parquet: {parquet_path}")
        print(f"  Metadata: {meta_path}")

        # Load embeddings (memory-mapped for efficiency)
        self.embeddings = np.memmap(embeddings_path, dtype=np.float16, mode="r")

        # Load metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta_info = json.load(f)

        # Reshape embeddings based on metadata
        n_vectors = self.meta_info["n_vectors"]
        dim = self.meta_info["dim"]
        self.embeddings = self.embeddings.reshape(n_vectors, dim)

        # Store file paths for chunked loading
        self.parquet_path = parquet_path

        # Load main metadata only once (if not already loaded)
        if self.main_metadata_df is None:
            print(f"  📊 Loading main metadata from: {self.config.MAIN_METADATA_PATH}")
            main_columns = [
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

            # Load with optimizations
            self.main_metadata_df = pd.read_parquet(
                self.config.MAIN_METADATA_PATH,
                columns=main_columns,
                use_threads=True,  # Enable multi-threading
            )
            print(f"✓ Loaded main metadata with {len(self.main_metadata_df)} documents")

        print(f"✓ Loaded {n_vectors} embeddings with dimension {dim}")
        print(f"📊 Will process data in chunks of {self.config.CHUNK_SIZE:,} documents")

    def load_chunk_data(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Load a specific chunk of parquet data using iterative row group reading"""
        print(f"   📊 Loading chunk data (rows {start_idx:,} to {end_idx:,})...")

        chunk_columns = [
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

        # Read the parquet file in row groups to find our target rows
        parquet_file = pq.ParquetFile(self.parquet_path)

        # Get metadata about the file
        num_row_groups = parquet_file.num_row_groups

        # Find which row groups contain our target rows
        current_row = 0
        target_row_groups = []

        for rg_idx in range(num_row_groups):
            rg_metadata = parquet_file.metadata.row_group(rg_idx)
            rg_num_rows = rg_metadata.num_rows
            rg_end_row = current_row + rg_num_rows

            # Check if this row group contains any of our target rows
            if (current_row < end_idx) and (rg_end_row > start_idx):
                target_row_groups.append(rg_idx)

            current_row = rg_end_row

            # If we've passed our target range, we can stop
            if current_row >= end_idx:
                break

        # Read only the necessary row groups
        if target_row_groups:
            print(f"   🔄 Reading {len(target_row_groups)} row groups...")
            table = parquet_file.read_row_groups(
                row_groups=target_row_groups, columns=chunk_columns, use_threads=True
            )
            df = table.to_pandas()

            # Calculate the offset within the read data
            # Find the actual start position within the loaded data
            rows_before_start = 0
            current_row = 0

            for rg_idx in range(num_row_groups):
                if rg_idx in target_row_groups:
                    break
                rg_metadata = parquet_file.metadata.row_group(rg_idx)
                current_row += rg_metadata.num_rows

            # Calculate relative positions within the loaded dataframe
            relative_start = max(0, start_idx - current_row)
            relative_end = min(len(df), end_idx - current_row)

            # Slice to get the exact chunk we want
            chunk_df = df.iloc[relative_start:relative_end].copy()

            # Clean up
            del df
            gc.collect()

        else:
            # Fallback: create empty dataframe with correct columns
            print(f"   ⚠ No data found in specified range, creating empty dataframe")
            chunk_df = pd.DataFrame(columns=chunk_columns)

        print(f"   ✓ Loaded chunk with {len(chunk_df):,} rows")
        return chunk_df

    def _preprocess_chunk_data(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """🚀 OPTIMIZED: Process a chunk of data and merge with metadata"""

        print(f"🔄 Processing chunk with {len(chunk_df):,} documents...")

        # Helper function to safely convert values and handle nulls (vectorized)
        def safe_value_series(series, default=""):
            """Vectorized version of safe_value for pandas Series"""
            return series.fillna(default).astype(str).where(series.notna(), default)

        def safe_int_series(series, default=0):
            """Safely convert series to int, replacing nulls with default"""
            return pd.to_numeric(series, errors="coerce").fillna(default).astype(int)

        # Convert doc_id to string for consistent matching
        print("   🔧 Converting data types...")
        chunk_df["doc_id"] = chunk_df["doc_id"].astype(str)

        # Perform LEFT JOIN to merge chunk data with main metadata
        print(f"   🔗 Merging {len(chunk_df):,} chunks with metadata...")
        start_time = time.time()

        merged_df = chunk_df.merge(
            self.main_metadata_df,
            left_on="doc_id",
            right_on="documentid",
            how="left",
            suffixes=("_chunk", "_main"),
        )

        # Identify missing documents (where merge didn't find a match)
        missing_mask = merged_df["documentid"].isna()
        chunk_missing_docs = merged_df[missing_mask]["doc_id"].unique().tolist()
        self.missing_docs.extend(chunk_missing_docs)

        if chunk_missing_docs:
            print(
                f"   ⚠ Found {len(chunk_missing_docs)} documents not in main metadata for this chunk"
            )

        # Pre-process all metadata fields with safe conversions (vectorized operations)
        print("   🔧 Pre-processing metadata fields...")

        # Fill missing document metadata with defaults
        merged_df["documentid"] = merged_df["documentid"].fillna(merged_df["doc_id"])
        merged_df["itemtype_name"] = merged_df["itemtype_name"].fillna("UNKNOWN")
        merged_df["itemtype_description"] = merged_df["itemtype_description"].fillna(
            "Unknown Document Type"
        )
        merged_df["birimadi"] = merged_df["birimadi"].fillna("Unknown")

        # Create pre-processed metadata columns (all vectorized)
        merged_df["proc_chunk_id"] = safe_value_series(merged_df["chunk_id"])
        merged_df["proc_source_path"] = safe_value_series(merged_df["source_path"])
        merged_df["proc_doc_checksum"] = safe_value_series(merged_df["doc_checksum"])
        merged_df["proc_mdtext"] = safe_value_series(merged_df["mdtext"])
        merged_df["proc_documentid"] = safe_value_series(merged_df["documentid"])
        merged_df["proc_itemtype_name"] = safe_value_series(merged_df["itemtype_name"])
        merged_df["proc_itemtype_description"] = safe_value_series(
            merged_df["itemtype_description"]
        )
        merged_df["proc_birimadi"] = safe_value_series(merged_df["birimadi"])
        merged_df["proc_karartarihi"] = safe_value_series(merged_df["karartarihi"])
        merged_df["proc_karartarihistr"] = safe_value_series(
            merged_df["karartarihistr"]
        )
        merged_df["proc_kesinlesmedurumu"] = safe_value_series(
            merged_df["kesinlesmedurumu"]
        )
        merged_df["proc_kararno"] = safe_value_series(merged_df["kararno"])
        merged_df["proc_esasno"] = safe_value_series(merged_df["esasno"])

        # Handle integer fields
        merged_df["proc_start_char"] = safe_int_series(merged_df["start_char"])
        merged_df["proc_end_char"] = safe_int_series(merged_df["end_char"])
        merged_df["proc_num_tokens"] = safe_int_series(merged_df["num_tokens"])
        merged_df["proc_esasnoyil"] = safe_int_series(merged_df["esasnoyil"])
        merged_df["proc_esasnosira"] = safe_int_series(merged_df["esasnosira"])
        merged_df["proc_kararnoyil"] = safe_int_series(merged_df["kararnoyil"])
        merged_df["proc_kararnosira"] = safe_int_series(merged_df["kararnosira"])

        # Create text previews (vectorized)
        text_lengths = merged_df["proc_mdtext"].str.len()
        merged_df["proc_text_preview"] = merged_df["proc_mdtext"].where(
            text_lengths <= 200, merged_df["proc_mdtext"].str[:200] + "..."
        )

        # Memory optimization: Use categorical for repeated values
        print("   💾 Optimizing memory usage...")
        for col in ["proc_itemtype_name", "proc_itemtype_description", "proc_birimadi"]:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].astype("category")

        # Performance timing
        end_time = time.time()
        preprocessing_time = end_time - start_time

        print(f"   ✅ Chunk preprocessing complete! {len(merged_df):,} records ready")
        print(f"   ⏱️  Processing took: {preprocessing_time:.2f} seconds")

        return merged_df

    def initialize_pinecone(self):
        """Initialize Pinecone connection and index"""
        if not self.config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        # Use gRPC client with optimized connection settings for better performance
        self.pc = PineconeGRPC(
            api_key=self.config.PINECONE_API_KEY, pool_threads=self.config.POOL_THREADS
        )

        # Check if index exists, create if not
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

    def save_missing_docs_info(self, file_range: str):
        """Save information about documents not found in main metadata"""
        missing_docs_info = {
            "file_range": file_range,
            "total_missing": len(self.missing_docs),
            "missing_doc_ids": self.missing_docs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "These doc_ids from chunk parquet were not found in main metadata parquet",
        }

        output_file = f"missing_docs_{file_range}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(missing_docs_info, f, indent=2, ensure_ascii=False)

        print(f"📋 Missing documents info saved to: {output_file}")
        print(
            f"   Total missing: {len(self.missing_docs)} out of {self.total_processed} total processed"
        )

    def prepare_vectors_for_chunk(
        self, merged_df: pd.DataFrame, start_idx: int
    ) -> Generator[List[Dict], None, None]:
        """🚀 OPTIMIZED: Prepare vectors in batches for a specific chunk"""

        batch = []
        total_rows = len(merged_df)

        print(f"🚀 Processing {total_rows:,} vectors in current chunk...")

        # Use iterrows with tqdm for progress tracking
        for row_idx, row in tqdm(
            merged_df.iterrows(), total=total_rows, desc="Preparing vectors"
        ):
            # Calculate the actual embedding index (start_idx + relative position)
            embedding_idx = start_idx + (row_idx - merged_df.index[0])

            # Get embedding vector (convert from float16 to float32 for Pinecone)
            vector_values = self.embeddings[embedding_idx].astype(np.float32).tolist()

            # Create unified metadata using pre-processed fields
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
                "mdtext": row["proc_mdtext"],  # Full text content
            }

            # Create vector record
            vector_record = {
                "id": f"doc_{row['doc_id']}_chunk_{row['chunk_id']}",
                "values": vector_values,
                "metadata": unified_metadata,
            }

            batch.append(vector_record)

            # Yield batch when it reaches desired size
            if len(batch) >= self.config.BATCH_SIZE:
                yield batch
                batch = []

        # Yield remaining vectors
        if batch:
            yield batch

    def upload_chunk_to_pinecone(self, vector_batches: List[List[Dict]]) -> int:
        """🚀 Upload a chunk of vectors to Pinecone"""

        uploaded_count = 0
        failed_batches = []

        print(f"📦 Uploading {len(vector_batches)} batches...")

        # Process batches in smaller groups to avoid overwhelming the API
        chunk_size = min(self.config.MAX_WORKERS, len(vector_batches))

        for chunk_start in range(0, len(vector_batches), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(vector_batches))
            batch_chunk = vector_batches[chunk_start:chunk_end]

            # Submit async requests for this chunk
            chunk_async_results = []
            for batch_idx, batch in enumerate(batch_chunk):
                try:
                    # Upload batch asynchronously
                    async_result = self.index.upsert(
                        vectors=batch,
                        namespace=self.config.NAMESPACE,
                        async_req=True,  # ASYNC upload for speed
                    )
                    chunk_async_results.append(
                        (chunk_start + batch_idx, async_result, len(batch))
                    )

                except Exception as e:
                    print(f"Failed to submit batch {chunk_start + batch_idx}: {e}")
                    failed_batches.append(chunk_start + batch_idx)

            # Wait for this chunk to complete
            for batch_idx, async_result, batch_size in chunk_async_results:
                try:
                    # For PineconeGRPC, we need to use result() method instead of get()
                    async_result.result()  # Wait for completion
                    uploaded_count += batch_size

                except Exception as e:
                    print(f"Failed to complete batch {batch_idx}: {e}")
                    failed_batches.append(batch_idx)

            # Small delay between chunks to avoid rate limiting
            if chunk_end < len(vector_batches):
                time.sleep(0.1)

        if failed_batches:
            print(f"⚠ Failed batches in this chunk: {failed_batches}")

        return uploaded_count

    def process_in_chunks(self, file_range: str):
        """Process and upload data in chunks to prevent RAM crashes"""

        # Get total number of vectors
        total_vectors = self.meta_info["n_vectors"]
        print(
            f"🚀 Processing {total_vectors:,} total vectors in chunks of {self.config.CHUNK_SIZE:,}"
        )

        total_uploaded = 0
        file_start_time = time.time()

        # Process data in chunks
        for chunk_start in range(0, total_vectors, self.config.CHUNK_SIZE):
            chunk_end = min(chunk_start + self.config.CHUNK_SIZE, total_vectors)
            chunk_num = (chunk_start // self.config.CHUNK_SIZE) + 1
            total_chunks = (
                total_vectors + self.config.CHUNK_SIZE - 1
            ) // self.config.CHUNK_SIZE

            print(f"\n{'='*60}")
            print(f"📦 Processing Chunk {chunk_num}/{total_chunks} for {file_range}")
            print(
                f"   Range: {chunk_start:,} to {chunk_end:,} ({chunk_end - chunk_start:,} documents)"
            )
            print(f"{'='*60}")

            chunk_start_time = time.time()

            # Step 1: Load chunk data
            chunk_df = self.load_chunk_data(chunk_start, chunk_end)

            # Step 2: Preprocess chunk
            merged_chunk_df = self._preprocess_chunk_data(chunk_df)

            # Step 3: Prepare vectors for this chunk
            print("🔄 Preparing vectors for upload...")
            vector_batches = list(
                self.prepare_vectors_for_chunk(merged_chunk_df, chunk_start)
            )

            # Step 4: Upload this chunk
            print(f"🚀 Uploading chunk {chunk_num}...")
            chunk_uploaded = self.upload_chunk_to_pinecone(vector_batches)
            total_uploaded += chunk_uploaded

            # Step 5: Clean up memory
            print("🧹 Cleaning up memory...")
            del chunk_df
            del merged_chunk_df
            del vector_batches
            gc.collect()  # Force garbage collection

            # Progress report
            chunk_time = time.time() - chunk_start_time
            file_time = time.time() - file_start_time
            overall_time = (
                time.time() - self.overall_start_time
                if self.overall_start_time
                else file_time
            )
            file_rate = total_uploaded / file_time if file_time > 0 else 0

            print(f"✅ Chunk {chunk_num} complete!")
            print(f"   Uploaded: {chunk_uploaded:,} vectors in {chunk_time:.1f}s")
            print(f"   File progress: {total_uploaded:,}/{total_vectors:,}")
            print(f"   File rate: {file_rate:.0f} vectors/sec")
            print(
                f"   Overall uploaded: {self.overall_total_uploaded + total_uploaded:,}"
            )

            self.total_processed = total_uploaded

        print(f"\n🎉 File {file_range} complete! Uploaded: {total_uploaded:,} vectors")
        return total_uploaded

    def run_upload_for_range(self, file_range: str):
        """Upload pipeline for a specific file range"""
        try:
            print(f"\n{'#'*80}")
            print(f"🚀 Processing file range: {file_range}")
            print(f"{'#'*80}")

            # Find files for this range
            embeddings_path, parquet_path, meta_path = self.find_files_for_range(
                file_range
            )

            # Load files for this range
            self.load_files(embeddings_path, parquet_path, meta_path)

            # Reset missing docs for this file range
            self.missing_docs = []

            # Process and upload in chunks
            total_uploaded = self.process_in_chunks(file_range)

            # Update overall total
            self.overall_total_uploaded += total_uploaded

            # Save missing docs info for this range
            if self.missing_docs:
                self.save_missing_docs_info(file_range)

            print(f"✅ Completed {file_range}: {total_uploaded:,} vectors uploaded")
            return total_uploaded

        except Exception as e:
            print(f"❌ Failed to process {file_range}: {e}")
            raise

    def run_upload_all_ranges(self):
        """Complete upload pipeline for all file ranges"""
        try:
            print("=" * 80)
            print("🚀 PINECONE UPLOAD PIPELINE - MULTIPLE FILE RANGES")
            print("=" * 80)

            # Initialize Pinecone once
            self.initialize_pinecone()

            # Set overall start time
            self.overall_start_time = time.time()

            # Process each file range
            for i, file_range in enumerate(self.config.FILE_RANGES, 1):
                print(
                    f"\n🔄 Processing range {i}/{len(self.config.FILE_RANGES)}: {file_range}"
                )

                try:
                    self.run_upload_for_range(file_range)
                except Exception as e:
                    print(f"⚠ Skipping {file_range} due to error: {e}")
                    continue

                # Clean up embeddings after each file to free memory
                if hasattr(self, "embeddings"):
                    del self.embeddings
                gc.collect()

            # Final verification
            print("\n🔍 Final verification...")
            time.sleep(5)  # Wait for indexing
            stats = self.index.describe_index_stats()
            print(f"📊 Final index stats: {stats}")

            # Overall summary
            overall_time = time.time() - self.overall_start_time
            overall_rate = (
                self.overall_total_uploaded / overall_time if overall_time > 0 else 0
            )

            print("\n" + "=" * 80)
            print("🎉 ALL UPLOADS COMPLETE!")
            print("=" * 80)
            print(f"📊 Total vectors uploaded: {self.overall_total_uploaded:,}")
            print(f"⏱️  Total time: {overall_time/60:.1f} minutes")
            print(f"🚀 Average rate: {overall_rate:.0f} vectors/sec")
            print("=" * 80)

            return self.overall_total_uploaded

        except Exception as e:
            print(f"Upload pipeline failed: {e}")
            raise


def test_query(index, query_text: str = "test query"):
    """Test the uploaded data with a sample query"""
    # This would require generating an embedding for the query text
    # You'll need to use the same embedding model (ytu-ce-cosmos/turkish-e5-large)
    print(f"Testing query: '{query_text}'")
    print(
        "Note: You'll need to generate an embedding for the query using the same model"
    )


if __name__ == "__main__":
    # Configuration
    config = Config()

    # Create uploader and run for all ranges
    uploader = PineconeUploader(config)
    uploader.run_upload_all_ranges()

    # Optional: Test the upload
    # test_query(uploader.index, "mahkeme kararı hakkında")
