from pipeline import RAGPipeline
from chunkers import *
from extractors import *
from stores import *

from dotenv import load_dotenv
load_dotenv()

pipeline = RAGPipeline(
    extractor=LLMIndexTimeExtractor(),
    chunker=SectionChunker(),
    store=ChromaStore()   # start local, swap to Pinecone later
)