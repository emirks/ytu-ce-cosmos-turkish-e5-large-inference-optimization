# Pinecone Upload and Query for Turkish Legal Documents

This project provides tools to upload Turkish legal document embeddings to Pinecone and perform semantic search queries.

## Overview

You have:
- **Embeddings**: Turkish E5 model embeddings stored as `.fp16.npy` memory-mapped files (1024 dimensions)
- **Parquet data**: Document chunks with metadata (columns: `['id', 'doc_id', 'chunk_id', 'start_char', 'end_char', 'mdtext', 'num_tokens', 'source_path', 'doc_checksum']`)
- **Metadata JSON**: Model configuration and embedding details

This solution uploads them to Pinecone with properly structured metadata for both YARGITAYKARARI and YERELHUKUK document types.

## Files Created

- `pinecone_upload.py` - Main upload script
- `query_pinecone.py` - Query and search script
- `requirements_pinecone.txt` - Python dependencies
- `env_example.txt` - Environment configuration template

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_pinecone.txt
```

### 2. Pinecone Account Setup

1. Sign up at [Pinecone](https://app.pinecone.io/)
2. Create a new API key
3. Note your preferred cloud region (e.g., `us-east-1`)

### 3. Environment Configuration

1. Copy `env_example.txt` to `.env`:
   ```bash
   cp env_example.txt .env
   ```

2. Edit `.env` with your Pinecone credentials:
   ```bash
   PINECONE_API_KEY=your_actual_api_key_here
   PINECONE_INDEX_NAME=yargi-ai-index
   PINECONE_ENVIRONMENT=us-east-1
   PINECONE_CLOUD=aws
   ```

### 4. File Structure

Ensure your files follow this pattern:
```
your_project/
├── embeddings_2m_3m.fp16.npy    # Your embeddings
├── chunks_2m_3m.parquet         # Your parquet data  
├── meta_2m_3m.json              # Your metadata
├── pinecone_upload.py           # Upload script
├── query_pinecone.py            # Query script
└── .env                         # Your credentials
```

The script will auto-detect files matching the patterns above.

## Usage

### Uploading to Pinecone

```bash
python pinecone_upload.py
```

**What it does:**
1. **Auto-detects** your embedding, parquet, and metadata files
2. **Loads embeddings** as memory-mapped numpy arrays (efficient for large files)
3. **Creates Pinecone index** with correct dimensions (1024D) if it doesn't exist
4. **Extracts metadata** from `source_path` to determine document types:
   - YARGITAYKARARI (Yargıtay Kararı)
   - YERELHUKUK (Yerel Hukuk Mahkemesi Kararı)
5. **Uploads in batches** of 100 vectors with rich metadata
6. **Validates upload** and shows index statistics

**Metadata Structure in Pinecone:**
```json
{
  "documentId": "71370900",
  "itemType": {
    "name": "YARGITAYKARARI", 
    "description": "Yargıtay Kararı"
  },
  "birimAdi": "12. Hukuk Dairesi",
  "esasNoYil": 2006,
  "esasNoSira": 10799,
  "kararNoYil": 2006, 
  "kararNoSira": 13163,
  "kararNo": "2006/13163",
  "esasNo": "2006/10799",
  "chunk_id": "123",
  "start_char": 0,
  "end_char": 500,
  "num_tokens": 128,
  "source_path": "path/to/original/file",
  "doc_checksum": "abc123",
  "text_preview": "First 200 characters of the text..."
}
```

### Querying the Index

```bash
python query_pinecone.py
```

**Query Examples:**

1. **General semantic search:**
   ```python
   results = querier.text_search("mahkeme kararı hukuki sorumluluk", top_k=5)
   ```

2. **Filter by document type:**
   ```python
   # Only Yargıtay decisions
   yargitay_filter = {"itemType.name": {"$eq": "YARGITAYKARARI"}}
   results = querier.text_search("ticaret hukuku", filter_dict=yargitay_filter)
   
   # Only local court decisions  
   local_filter = {"itemType.name": {"$eq": "YERELHUKUK"}}
   results = querier.text_search("asliye mahkemesi", filter_dict=local_filter)
   ```

3. **Filter by year:**
   ```python
   # Decisions from 2020 onwards
   year_filter = {"kararNoYil": {"$gte": 2020}}
   results = querier.text_search("tazminat davası", filter_dict=year_filter)
   ```

4. **Combined filters:**
   ```python
   # Yargıtay decisions from specific department after 2020
   complex_filter = {
     "$and": [
       {"itemType.name": {"$eq": "YARGITAYKARARI"}},
       {"kararNoYil": {"$gte": 2020}},
       {"birimAdi": {"$contains": "Hukuk Dairesi"}}
     ]
   }
   ```

## Key Features

### 🚀 **Performance Optimized**
- Uses Pinecone gRPC client for faster uploads
- Memory-mapped numpy arrays for efficient memory usage
- Batched uploads (100 vectors per batch)
- Parallel processing support

### 🎯 **Smart Metadata Extraction**
- Automatically detects document types from file paths
- Extracts court names, case numbers, and years
- Handles both YARGITAYKARARI and YERELHUKUK formats
- Preserves original chunk-level metadata

### 🔍 **Advanced Querying**
- Turkish text-to-embedding conversion using same E5 model
- Rich metadata filtering capabilities
- Support for complex boolean filters
- Formatted result display

### 📊 **Monitoring & Validation**
- Upload progress tracking with tqdm
- Automatic validation of file consistency
- Index statistics and health checks
- Error handling and retry logic

## Troubleshooting

### Common Issues

1. **"PINECONE_API_KEY not set"**
   - Make sure your `.env` file exists and contains the correct API key
   - Load environment: `python -c "import os; print(os.getenv('PINECONE_API_KEY'))"`

2. **"Could not auto-detect files"**
   - Ensure your files match the expected patterns: `embeddings_*.fp16.npy`, `chunks_*.parquet`, `meta_*.json`
   - Or manually specify paths in the config

3. **"Dimension mismatch"**
   - Verify your embeddings are 1024-dimensional (Turkish E5 model)
   - Check the metadata JSON file for correct dimension info

4. **"Upload failed" / Rate limiting**
   - Pinecone free tier has limits; the script includes retry logic
   - Consider using smaller batch sizes or adding delays

5. **"Out of memory"**
   - Memory-mapped arrays should handle large files efficiently
   - If issues persist, process files in smaller chunks

### Performance Tips

1. **For large datasets (1M+ vectors):**
   - Use the gRPC client (already enabled)
   - Consider parallel upload with `async_req=True`
   - Monitor Pinecone usage limits

2. **For faster queries:**
   - Use specific metadata filters to reduce search space
   - Cache the embedding model for multiple queries
   - Consider using namespaces for data organization

## Cost Optimization

### Pinecone Costs
- **Free tier**: 1 index, 100K vectors, 1GB storage
- **Paid plans**: Based on vector count and query volume
- See [Pinecone pricing](https://www.pinecone.io/pricing/)

### Recommendations
1. **Start with free tier** for testing/development
2. **Use metadata filters** to reduce query costs
3. **Monitor usage** via Pinecone dashboard
4. **Consider namespaces** for multi-tenant scenarios

## Advanced Configuration

### Custom File Paths
```python
# In pinecone_upload.py
config.EMBEDDINGS_PATH = "custom/path/to/embeddings.npy"
config.PARQUET_PATH = "custom/path/to/data.parquet" 
config.META_PATH = "custom/path/to/meta.json"
```

### Batch Size Tuning
```python
# Adjust based on your data size and memory
config.BATCH_SIZE = 50    # Smaller batches for large metadata
config.BATCH_SIZE = 200   # Larger batches for minimal metadata
```

### Index Configuration
```python
# Different distance metrics
config.DISTANCE_METRIC = "euclidean"  # or "dotproduct"

# Different cloud regions
config.PINECONE_ENVIRONMENT = "eu-west1"  # European region
```

## Example Output

### Upload Process:
```
=== Pinecone Upload Pipeline ===
Loading files:
  Embeddings: embeddings_2m_3m.fp16.npy
  Parquet: chunks_2m_3m.parquet
  Metadata: meta_2m_3m.json
✓ Loaded 1000000 embeddings with dimension 1024
✓ Loaded parquet data with 1000000 rows
Using existing index: yargi-ai-index
Starting upload to Pinecone index: yargi-ai-index
Preparing vectors: 100%|██████████| 1000000/1000000 [05:23<00:00, 3094.12it/s]
Uploaded 1000 vectors (10 batches)
Uploaded 2000 vectors (20 batches)
...
✓ Upload complete! Total vectors uploaded: 1000000
Index stats: {'total_vector_count': 1000000, 'namespaces': {'': {'vector_count': 1000000}}}
=== Upload Complete ===
```

### Query Results:
```
=== Search Results (3 matches) ===

--- Result 1 ---
ID: doc_123456_chunk_789
Score: 0.8542
Document ID: 123456
Type: YARGITAYKARARI - Yargıtay Kararı
Department: 12. Hukuk Dairesi
Case Number: 2006/10799
Decision Number: 2006/13163
Chunk: 789 (1250-1850)
Text: Mahkemenin vermiş olduğu karar, hukuki açıdan değerlendirildiğinde...
```

## Support

For issues:
1. Check the troubleshooting section above
2. Verify your Pinecone account and API key
3. Ensure file formats match expected structure
4. Review error messages for specific guidance

## References

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Turkish E5 Model](https://huggingface.co/ytu-ce-cosmos/turkish-e5-large)
- [Pinecone Python SDK](https://docs.pinecone.io/reference/python-sdk) 