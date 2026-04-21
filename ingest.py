"""
ingest.py
Run this ONCE (or on a schedule) to populate the vector DB.
Never called at query time — no ArXiv API calls happen during search.

Usage:
    python ingest.py
    python ingest.py --queries "CNN CIFAR" "transformer NLP" "GAN image synthesis"
"""

import argparse
from dotenv import load_dotenv
load_dotenv()

# ── Pick your components (same as main.py) ─────────────────────────────────
from extractors import LLMIndexTimeExtractor   # Option A
# from extractors.llm_query_time import LLMQueryTimeExtractor  # Option B

from chunkers import SectionChunker           # Option B
# from chunkers.implementations import ParagraphChunker        # Option C
# from chunkers.implementations import HierarchicalChunker     # Option D

from stores import ChromaStore                # Option D (local)
# from stores.implementations import PineconeStore             # Option A (cloud)
# from stores.implementations import WeaviateStore             # Option B (hybrid)

from pipeline import RAGPipeline

# ── Default queries to ingest if none are passed via CLI ───────────────────
DEFAULT_QUERIES = [
    "CNN image classification CIFAR accuracy",
    "transformer attention mechanism NLP",
    "GAN generative adversarial network image synthesis",
    "BERT language model fine-tuning",
    "object detection YOLO real-time",
    "transformer models", "attention models",
    "time series", 
    "nlp", 
    "artificial intelligence",
    "deep learning", 
    "machine learning"
]


def main():
    
    parser = argparse.ArgumentParser(description="Ingest ArXiv papers into the vector DB.")
    parser.add_argument(
        "--queries", nargs="+",
        default=DEFAULT_QUERIES,
        help="ArXiv search queries to ingest",
    )
    parser.add_argument(
        "--max-results", type=int, default=10,
        help="Max papers to fetch per query (default: 10)",
    )
    args = parser.parse_args()



    pipeline = RAGPipeline(
        extractor=LLMIndexTimeExtractor(),
        chunker=SectionChunker(),
        store=ChromaStore(),
        max_arxiv_results=args.max_results,
    )

    total_chunks = 0
    for query in args.queries:
        print(f"\n{'='*60}")
        chunks_added = pipeline.ingest_2(query, segment_size = 3000)
        total_chunks += chunks_added

    print(f"\n{'='*60}")
    print(f"[Ingest] Done. Total new chunks indexed: {total_chunks}")
    print("[Ingest] The vector DB is now ready for querying.")


if __name__ == "__main__":
    main()