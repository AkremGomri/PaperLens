from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import arxiv
import fitz
from AbstractClasses.BaseExtractor import BaseExtractor
from AbstractClasses.BaseChunker import BaseChunker
from AbstractClasses.BaseStore import BaseStore
from utils.file_writer import save_and_diff, write_to_file

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant helping data scientists find relevant papers.
Use ONLY the provided context to answer the question.
Always cite the paper title and ArXiv ID when referencing a result.
If the context doesn't contain enough information, say so honestly."""),
    ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
])

class RAGPipeline:
    def __init__(
        self,
        extractor: BaseExtractor,
        chunker: BaseChunker,
        store: BaseStore,
        llm_model: str = "gpt-4o-mini",
        max_arxiv_results: int = 10,
    ):
        self.extractor = extractor
        self.chunker = chunker
        self.store = store
        self.max_inject_results = max_arxiv_results
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.qa_chain = QA_PROMPT | self.llm

        # If using query-time extraction, we detect it here
        from extractors import LLMQueryTimeExtractor
        self._query_time_extraction = isinstance(extractor, LLMQueryTimeExtractor)

    def ingest(self, arxiv_query: str) -> int:
        """
        Fetch papers from ArXiv, extract metadata, chunk, and index.
        Skips papers already present in the vector DB (deduplication by arxiv_id).
        Returns the number of NEW chunks indexed.
        """
        #print(f"[Pipeline] Fetching papers for: '{arxiv_query}'")
        documents = arxiv.Search(
            query = arxiv_query,
            max_results = self.max_inject_results
            ).results()
        #print(f"[Pipeline] Fetched {len(documents)} papers")        
 
        # Deduplication — skip papers already in the vector DB
        already_indexed = self.store.get_indexed_ids()
        print(type(already_indexed))
        print(type(documents))
        new_documents = [
            doc for doc in documents
            if doc.entry_id.split("/")[-1] not in already_indexed
        ]
        skipped = len(list(documents)) - len(new_documents)
        if skipped:
            print(f"[Pipeline] Skipping {skipped} already-indexed papers")
        if not new_documents:
            print("[Pipeline] Nothing new to index.")
            return 0
 
        all_chunks = []
        for doc in new_documents:
 
            # Option A: extract metadata at index time and store it in metadata
            if not self._query_time_extraction:
                print(f"[Pipeline] Extracting metadata for: {doc.title[:60]}")
                extracted = self.extractor.extract(doc)
                doc.metadata["models"]   = extracted.get("models", [])
                doc.metadata["datasets"] = extracted.get("datasets", [])
                import json
                doc.metadata["metrics"]  = json.dumps(extracted.get("metrics", []))
            elif self._query_time_extraction:
                print(f"[Pipeline] Skipping index-time extraction for: {doc.title[:60]}")
 
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
 
        print(f"[Pipeline] Indexing {len(all_chunks)} new chunks")
        self.store.index(all_chunks)
        return len(all_chunks)

    def ingest_2(self, arxiv_query: str, segment_size: int) -> int:
        """
        Fetch papers from a local directory, rewrite the text using an llm page by page, chunk, and index.
        Skips papers already present in the vector DB (deduplication by arxiv_id).
        Returns the number of NEW chunks indexed.
        """

        Data_Folder_Path = "Data/"
        Output_Folder_Path = "Data_fixed/"
        documents_path = os.listdir(Data_Folder_Path)
        # Deduplication — skip papers already in the vector DB
        already_indexed = self.store.get_indexed_ids()
        print(type(already_indexed))
        print(type(documents_path))
        new_documents_path = [
            doc for doc in documents_path
            if doc.split(".")[0] not in already_indexed
        ]
        skipped = len(list(documents_path)) - len(new_documents_path)
        if skipped:
            print(f"[Pipeline] Skipping {skipped} already-indexed papers")
        if not new_documents_path:
            print("[Pipeline] Nothing new to index.")
            return 0
        all_chunks = []
        for doc_name in new_documents_path:
            doc_path = os.path.join(Data_Folder_Path, doc_name)
            doc = fitz.open(doc_path)
            for i in range(0, doc.page_count):
                page = doc.load_page(i)
                text = page.get_text()  # Use only the first 3000 chars to keep costs low
                segments = [text[i:i + segment_size] for i in range(0, len(text), segment_size)]
                for j, segment in enumerate(segments):
                    print("doc_path:", doc_path)
                    print("j:", j)
                    print("text before rewriting:", segment)
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", 
                        "You are a research scientific paper reader and fixer. "
                        "Your task is to clean formatting issues while preserving the original text.\n\n"
                        "Rules:\n"
                        "- Preserve all original words exactly.\n"
                        "- Fix broken line breaks.\n"
                        "- Reconstruct inline equations into readable latex format when possible.\n"
                        "- Keep mathematical expressions clear (e.g., 'R = sum(...)').\n"
                        "- create tables if the text suggests tabular data.\n"
                        "- Do NOT summarize or remove content.\n"
                        "- Return only the cleaned text."
                        ),
                        ("user", "{segment}")
                    ])
                    rewrite_chain = prompt | self.llm
                    llm_response = rewrite_chain.invoke({"segment": segment})
                    rewritten = llm_response.content.strip()
                    print("rewritten text:", rewritten)

                    save_and_diff(
                        original_text=segment, 
                        fixed_text=rewritten, 
                        doc_name=doc_name, 
                        page_num=i if j == 0 else None
                        )
                    write_to_file(
                        text=rewritten,
                        file_path=os.path.join(Output_Folder_Path, doc_name)
                    )
                    
                    chunk = Document(
                        page_content=rewritten,
                        metadata={"source": doc_path, "page": i}
                    )
                    print("--------------------------------------------------")
                    all_chunks.append(chunk)
        print(f"[Pipeline] Indexing {len(all_chunks)} new chunks")
        self.store.index(all_chunks)
        return len(all_chunks)

    def search(self, query, filters=None):
        return self.store.query(query, filters)

    def answer(self, query, filters=None):
        docs = self.search(query, filters)
        # pass docs to an LLM and return the answer
        ...