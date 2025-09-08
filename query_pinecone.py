#!/usr/bin/env python3
"""
Pinecone Query Script for Turkish Legal Documents
Query the uploaded embeddings using text queries and metadata filters
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel
from pinecone.grpc import PineconeGRPC
from pinecone import Pinecone


class TurkishEmbeddingModel:
    """Turkish E5 model for generating query embeddings"""

    def __init__(self, model_id: str = "ytu-ce-cosmos/turkish-e5-large"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Turkish E5 model"""
        print(f"Loading model: {self.model_id} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        print("✓ Model loaded successfully")

    def encode(self, text: str, prefix: str = "passage: ") -> np.ndarray:
        """Generate embedding for a text query"""
        # Add prefix for E5 model (as used in original embedding script)
        input_text = f"{prefix}{text}"

        # Tokenize
        tokens = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        # Generate embedding
        with torch.inference_mode():
            outputs = self.model(**tokens)

            # Mean pooling
            attention_mask = tokens["attention_mask"]
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            summed = (outputs.last_hidden_state * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-6)
            embedding = summed / counts

            # Normalize
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy().flatten().astype(np.float32)


class PineconeQuerier:
    """Query Pinecone index for Turkish legal documents"""

    def __init__(self, api_key: str, index_name: str):
        self.pc = PineconeGRPC(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.embedding_model = None

    def load_embedding_model(self):
        """Load the Turkish embedding model for queries"""
        if self.embedding_model is None:
            self.embedding_model = TurkishEmbeddingModel()

    def text_search(
        self,
        query_text: str,
        top_k: int = 10,
        namespace: str = "",
        filter_dict: Optional[Dict] = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> Dict:
        """
        Search using text query

        Args:
            query_text: Turkish text to search for
            top_k: Number of results to return
            namespace: Pinecone namespace to search in
            filter_dict: Metadata filters (e.g., {"itemType.name": {"$eq": "YARGITAYKARARI"}})
            include_metadata: Whether to include metadata in results
            include_values: Whether to include embedding values in results
        """

        # Generate embedding for query
        if self.embedding_model is None:
            self.load_embedding_model()

        query_embedding = self.embedding_model.encode(query_text)

        # Perform search
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            include_metadata=include_metadata,
            include_values=include_values,
        )

        return results

    def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        namespace: str = "",
        filter_dict: Optional[Dict] = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> Dict:
        """Search using pre-computed embedding vector"""

        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            include_metadata=include_metadata,
            include_values=include_values,
        )

        return results

    def get_index_stats(self) -> Dict:
        """Get statistics about the index"""
        return self.index.describe_index_stats()

    def fetch_by_ids(self, ids: List[str], namespace: str = "") -> Dict:
        """Fetch specific documents by their IDs"""
        return self.index.fetch(ids=ids, namespace=namespace)


def format_search_results(results: Dict, max_text_length: int = 200) -> None:
    """Pretty print search results"""

    print(f"\n=== Search Results ({len(results['matches'])} matches) ===")

    for i, match in enumerate(results["matches"], 1):
        print(f"\n--- Result {i} ---")
        print(f"ID: {match['id']}")
        print(f"Score: {match['score']:.4f}")

        if "metadata" in match:
            metadata = match["metadata"]

            # Document info
            if "documentId" in metadata:
                print(f"Document ID: {metadata['documentId']}")

            # Document type
            if "itemType" in metadata and isinstance(metadata["itemType"], dict):
                item_type = metadata["itemType"]
                print(
                    f"Type: {item_type.get('name', 'N/A')} - {item_type.get('description', 'N/A')}"
                )

            # Court/Department info
            if "birimAdi" in metadata and metadata["birimAdi"]:
                print(f"Department: {metadata['birimAdi']}")

            # Case numbers
            if "esasNo" in metadata and metadata["esasNo"]:
                print(f"Case Number: {metadata['esasNo']}")
            if "kararNo" in metadata and metadata["kararNo"]:
                print(f"Decision Number: {metadata['kararNo']}")

            # Chunk info
            if "chunk_id" in metadata:
                print(
                    f"Chunk: {metadata['chunk_id']} ({metadata.get('start_char', 'N/A')}-{metadata.get('end_char', 'N/A')})"
                )

            # Text preview
            if "text_preview" in metadata:
                text = metadata["text_preview"]
                if len(text) > max_text_length:
                    text = text[:max_text_length] + "..."
                print(f"Text: {text}")


def main():
    """Example usage of the query system"""

    # Configuration
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "yargi-ai-index")

    if not api_key:
        print("Error: PINECONE_API_KEY environment variable not set")
        return

    # Initialize querier
    querier = PineconeQuerier(api_key, index_name)

    # Example queries
    print("=== Turkish Legal Document Search ===")

    # 1. General text search
    print("\n1. General search:")
    results = querier.text_search("mahkeme kararı hukuki sorumluluk", top_k=5)
    format_search_results(results)

    # 2. Search with document type filter (only Yargıtay decisions)
    print("\n\n2. Search only Yargıtay decisions:")
    yargitay_filter = {"itemType.name": {"$eq": "YARGITAYKARARI"}}
    results = querier.text_search(
        "ticaret hukuku sözleşme ihlali", top_k=3, filter_dict=yargitay_filter
    )
    format_search_results(results)

    # 3. Search with year filter
    print("\n\n3. Search decisions from 2020 onwards:")
    year_filter = {"kararNoYil": {"$gte": 2020}}
    results = querier.text_search("tazminat davası", top_k=3, filter_dict=year_filter)
    format_search_results(results)

    # 4. Search local court decisions
    print("\n\n4. Search local court decisions:")
    local_filter = {"itemType.name": {"$eq": "YERELHUKUK"}}
    results = querier.text_search(
        "asliye mahkemesi karar", top_k=3, filter_dict=local_filter
    )
    format_search_results(results)

    # 5. Get index statistics
    print("\n\n=== Index Statistics ===")
    stats = querier.get_index_stats()
    print(f"Total vectors: {stats.get('total_vector_count', 'N/A')}")
    if "namespaces" in stats:
        for namespace, info in stats["namespaces"].items():
            ns_name = namespace if namespace else "(default)"
            print(f"Namespace '{ns_name}': {info.get('vector_count', 'N/A')} vectors")


if __name__ == "__main__":
    main()
