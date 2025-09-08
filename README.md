# Turkish E5 Large Inference Optimization

A comprehensive toolkit for optimizing Turkish language model inference, featuring embedding generation, vector indexing, and efficient querying capabilities.

## Project Structure

```
├── src/                    # Main source code
│   ├── embedding/         # Embedding generation and optimization
│   ├── indexing/          # Vector database indexing (FAISS, Pinecone)
│   ├── querying/          # Query and search functionality
│   ├── data_processing/   # Data processing and validation
│   ├── optimization/      # Model optimization (TensorRT, ONNX)
│   └── utils/             # Utility functions and helpers
├── scripts/               # Standalone scripts and tools
├── config/                # Configuration files
├── data/                  # Data files and datasets
│   ├── raw/              # Raw input data
│   ├── processed/        # Processed data files
│   ├── embeddings/       # Generated embeddings
│   └── indices/          # Built indices
├── docs/                  # Documentation
├── tests/                 # Test files
├── notebooks/             # Jupyter notebooks
└── requirements/          # Environment-specific requirements
```

## Key Components

### Embedding Generation (`src/embedding/`)
- **ytu_embed.py**: Main embedding generation using Turkish E5 Large model
- **embed_trt_simple.py**: TensorRT optimized embedding generation
- **batch_embed.py**: Batch processing for large datasets

### Vector Indexing (`src/indexing/`)
- **build_faiss_index.py**: Build FAISS indices for fast similarity search
- **pinecone_upload.py**: Upload embeddings to Pinecone vector database
- **pinecone_upload_memory_optimized.py**: Memory-efficient Pinecone upload

### Querying (`src/querying/`)
- **query_faiss.py**: Query FAISS indices
- **query_pinecone.py**: Query Pinecone database
- **pinecone_query.py**: Advanced Pinecone querying

### Data Processing (`src/data_processing/`)
- **make_chunks.py**: Chunk documents for processing
- **validate_corpus.py**: Validate document corpus
- **validate_parquets.py**: Validate parquet files
- **analyze_parquets.py**: Analyze parquet file structure

### Model Optimization (`src/optimization/`)
- **optimize_trt_model.py**: TensorRT model optimization
- **optimize_model.py**: General model optimization

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up configuration:
   ```bash
   cp config/env_example.txt .env
   # Edit .env with your API keys and paths
   ```

## Usage

### Generate Embeddings
```bash
python src/embedding/ytu_embed.py
```

### Build FAISS Index
```bash
python src/indexing/build_faiss_index.py
```

### Query System
```bash
python src/querying/query_faiss.py "your query here"
```

## Documentation

- [Pinecone Setup Guide](docs/README_PINECONE.md)
- [Usage Guide](docs/USAGE.md)
