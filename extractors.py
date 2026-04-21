import json
from AbstractClasses.BaseExtractor import BaseExtractor
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Then each option is just a concrete implementation
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research paper metadata extractor.
Given the text of a research paper, extract the following in JSON only — no preamble, no markdown:
{{
  "models":   ["list of ML model names or architectures mentioned, e.g. ResNet-50, BERT, CNN"],
  "metrics":  [{{"name": "metric name", "value": <float or null>, "dataset": "dataset name or null"}}],
  "datasets": ["list of dataset names mentioned"]
}}
If a field is not found, return an empty list."""),
    ("human", "Paper text (first 3000 chars):\n\n{text}"),
])
 
 
class LLMIndexTimeExtractor(BaseExtractor):
    """
    Option A: Extract metadata with an LLM at indexing time.
    Result is stored directly in the Document metadata.
    """
 
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.chain = EXTRACTION_PROMPT | self.llm
 
    def extract(self, document: Document) -> dict:
        # Use only the first 3000 chars to keep costs low
        print(dir(document))
        text_sample = document.summary
        print(f"text_sample: {text_sample}...")  # Debug: show the text being sent to the LLM
 
        try:
            response = self.chain.invoke({"text": text_sample})
            raw = response.content.strip()
            # Strip markdown code fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            print(f"[LLMIndexTimeExtractor] Extraction failed: {e}")
            return {"models": [], "metrics": [], "datasets": []}
        
class LLMQueryTimeExtractor(BaseExtractor):
    """
    Option B: Extract metadata at query time from retrieved chunks.
    No metadata stored at index time — extraction happens on the fly.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.chain = EXTRACTION_PROMPT | self.llm

    def extract(self, document: Document) -> dict:
        """
        For query-time extraction, this is called on retrieved chunks,
        not on the full paper. The pipeline passes a Document whose
        page_content is the concatenation of retrieved chunks.
        """
        try:
            response = self.chain.invoke({"chunks": document.page_content[:4000]})
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            print(f"[LLMQueryTimeExtractor] Extraction failed: {e}")
            return {"models": [], "metrics": [], "datasets": []}

    def extract_from_chunks(self, chunks: list[Document]) -> dict:
        """Convenience: extract from a list of retrieved Documents."""
        combined_text = "\n\n---\n\n".join(c.page_content for c in chunks)
        synthetic_doc = Document(page_content=combined_text)
        return self.extract(synthetic_doc)   # Option B