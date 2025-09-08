#!/usr/bin/env python3
"""
Simple Pinecone Query Script
Uses the same Turkish E5 model to embed queries and search the uploaded index
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pinecone.grpc import PineconeGRPC

# Configuration - should match your upload script
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "ytu-ce-cosmos-turkish-e5-large-embeddings"
MODEL_ID = "ytu-ce-cosmos/turkish-e5-large"
PREFIX = "query: "  # Note: different prefix for queries (query: vs passage:)
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16


class PineconeQueryEngine:
    def __init__(self):
        """Initialize the query engine with model and Pinecone connection"""
        print("🔄 Initializing Pinecone Query Engine...")

        # Load the same model used for embeddings
        print(f"📥 Loading model: {MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        self.model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
        self.model.eval()

        # Initialize Pinecone connection
        print(f"🔗 Connecting to Pinecone index: {PINECONE_INDEX_NAME}")
        self.pc = PineconeGRPC(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

        print("✅ Query engine ready!")

    def mean_pool(self, last_hidden_state, attention_mask):
        """Mean pooling function - same as used in embedding creation"""
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-6)
        return summed / counts

    def embed_query(self, query_text: str) -> np.ndarray:
        """Embed a query using the same method as document embedding"""
        # Add query prefix (different from passage prefix used in upload)
        prefixed_query = f"{PREFIX}{query_text}"

        # Tokenize
        tokens = self.tokenizer(
            prefixed_query,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        # Move to device
        input_ids = tokens["input_ids"].to(DEVICE)
        attention_mask = tokens["attention_mask"].to(DEVICE)

        # Generate embedding
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if DEVICE == "cuda" else "cpu", dtype=DTYPE
        ):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = self.mean_pool(outputs.last_hidden_state, attention_mask)
            # Normalize (same as in upload)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        # Convert to numpy float32 (Pinecone requirement)
        return embedding.cpu().numpy().astype(np.float32)[0]

    def search(self, query_text: str, top_k: int = 5, include_metadata: bool = True):
        """Search for similar documents"""
        print(f"🔍 Searching for: '{query_text}'")

        # Generate query embedding
        query_vector = self.embed_query(query_text)

        # Search in Pinecone
        results = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=include_metadata,
            include_values=False,  # We don't need the actual vectors back
        )

        return results

    def format_results(self, results, query_text: str):
        """Format and print search results"""
        print(f"\n📊 Search Results for: '{query_text}'")
        print("=" * 80)

        if not results.matches:
            print("❌ No results found")
            return

        for i, match in enumerate(results.matches, 1):
            print(f"\n🔹 Result {i} (Score: {match.score:.4f})")
            print(f"   ID: {match.id}")

            if match.metadata:
                # Display key metadata
                doc_id = match.metadata.get("documentId", "N/A")
                item_type = match.metadata.get("itemTypeName", "N/A")
                birim = match.metadata.get("birimAdi", "N/A")
                karar_tarihi = match.metadata.get("kararTarihi", "N/A")

                print(f"   Document ID: {doc_id}")
                print(f"   Type: {item_type}")
                print(f"   Birim: {birim}")
                print(f"   Karar Tarihi: {karar_tarihi}")

                # Show text preview (truncated)
                text = match.metadata.get("mdtext", "")

                print(f"   Text: {text}")

            print("-" * 60)

    def interactive_search(self):
        """Interactive search mode"""
        print("\n🎯 Interactive Search Mode")
        print("Type your queries (or 'quit' to exit)")
        print("=" * 50)

        while True:
            try:
                query = input("\n🔍 Query: ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not query:
                    continue

                # Search and display results
                results = self.search(query, top_k=5)
                self.format_results(results, query)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function with example queries"""
    # Initialize query engine
    engine = PineconeQueryEngine()

    # Example queries
    example_queries = [
        "mahkeme kararı hakkında",
        "boşanma davası",
        "miras hukuku",
        "iş hukuku",
        "ceza davası",
    ]

    print("\n🎯 Running example queries...")
    for query in example_queries:
        results = engine.search(query, top_k=10)
        engine.format_results(results, query)
        print("\n" + "=" * 80 + "\n")

    # Start interactive mode
    engine.interactive_search()


if __name__ == "__main__":
    main()
