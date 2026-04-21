
import re
from AbstractClasses.BaseChunker import BaseChunker
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Common research paper section headers
SECTION_PATTERN = re.compile(
    r"\n(?=(?:abstract|introduction|related work|background|methodology|"
    r"method|approach|model|architecture|experiments?|results?|"
    r"evaluation|discussion|conclusion|references?|appendix)\b)",
    re.IGNORECASE,
)

class fixedSizeWithOverlapChunker(BaseChunker):
    def chunk(self, source_text: str, chunk_size: int, overlap_fraction: float) -> List[str]:
        """
        Splits a given text into chunks of a fixed size with a specified overlap fraction between consecutive chunks.

        Parameters:
        - source_text (str): The input text to be split into chunks.
        - chunk_size (int): The number of words each chunk should contain.
        - overlap_fraction (float): The fraction of the chunk size that should overlap with the adjacent chunk.
        For example, an overlap_fraction of 0.2 means 20% of the chunk size will be used as overlap.

        Returns:
        - List[str]: A list of chunks (each a string) where each chunk might overlap with its adjacent chunk.
        """

        # Split the text into individual words
        text_words = source_text.split()
        
        # Calculate the number of words to overlap between consecutive chunks
        overlap_int = int(chunk_size * overlap_fraction)
        
        # Initialize a list to store the resulting chunks
        chunks = []
        
        # Iterate over text in steps of chunk_size to create chunks
        for i in range(0, len(text_words), chunk_size):
            # Determine the start and end indices for the current chunk,
            # taking into account the overlap with the previous chunk
            chunk_words = text_words[max(i - overlap_int, 0): i + chunk_size]
            
            # Join the selected words to form a chunk string
            chunk = " ".join(chunk_words)
            
            # Append the chunk to the list of chunks
            chunks.append(chunk)
        
        # Return the list of chunks
        return chunks
    
class ParagraphChunker(BaseChunker):
    def chunk(self, source_text: str) -> List[str]:
        return source_text.split("\n\n")
    
class SectionChunker(BaseChunker):
    def chunk(self, source_text: str) -> List[str]:
            """
        Option B: Splits papers on section boundaries (Abstract, Introduction,
        Methods, Results, etc.). Keeps evaluation scores near their context.

        Falls back to RecursiveCharacterTextSplitter if no sections are detected.
    """

    def __init__(self, max_chunk_size: int = 2000, chunk_overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, document: Document) -> list[Document]:
        sections = SECTION_PATTERN.split(document.page_content)

        # If no sections found, fall back to recursive splitting
        if len(sections) <= 1:
            return self.fallback_splitter.split_documents([document])

        chunks = []
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # If a section is still too large, split it further
            if len(section) > self.max_chunk_size:
                sub_docs = self.fallback_splitter.split_documents([
                    Document(page_content=section, metadata=document.metadata)
                ])
                chunks.extend(sub_docs)
            else:
                chunks.append(Document(
                    page_content=section,
                    metadata={**document.metadata, "section_index": i},
                ))

        return chunks
class MixedChunker(BaseChunker):
    def chunk(self, source_text: str) -> List[str]:
        """
        Splits the given source_text into chunks using a mix of fixed-size and variable-size chunking.
        It first splits the text by Asciidoc markers and then processes the chunks to ensure they are 
        of appropriate size. Smaller chunks are merged with the next chunk, and larger chunks can be 
        further split at the middle or specific markers within the chunk.

        Args:
        - source_text (str): The text to be chunked.

        Returns:
        - list: A list of text chunks.
        """

        # Split the text by Asciidoc marker
        chunks = source_text.split("\n==")

        # Chunking logic
        new_chunks = []
        chunk_buffer = ""
        min_length = 25

        for chunk in chunks:
            new_buffer = chunk_buffer + chunk  # Create new buffer
            new_buffer_words = new_buffer.split(" ")  # Split into words
            if len(new_buffer_words) < min_length:  # Check whether buffer length is too small
                chunk_buffer = new_buffer  # Carry over to the next chunk
            else:
                new_chunks.append(new_buffer)  # Add to chunks
                chunk_buffer = ""

        if len(chunk_buffer) > 0:
            new_chunks.append(chunk_buffer)  # Add last chunk, if necessary

        return new_chunks

class ContextAwareChunker(BaseChunker): ...            # Option B

class HierarchicalChunker(BaseChunker): ...       # Option D