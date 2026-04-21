import os
import arxiv

DOWNLOAD_DIR = "Data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

queries = ["transformer models", "attention models","time series", "nlp", "artificial intelligence","deep learning", "machine learning"]

for i, query in enumerate(queries):
    print(f"Processing query {i+1}/{len(queries)}: '{query}'")
    search = arxiv.Search(
        query=query,
        max_results=1000
    )

    for result in search.results():
        paper_id = result.entry_id.split("/")[-1]

        # 🔍 check if any file starts with this ID
        already_exists = any(
            f.startswith(paper_id) and f.endswith(".pdf")
            for f in os.listdir(DOWNLOAD_DIR)
        )

        if already_exists:
            print(f"Already exists: {paper_id}")
            continue

        # 📥 download with full naming style
        result.download_pdf(dirpath=DOWNLOAD_DIR)
        print(f"Just downloaded: {paper_id}")