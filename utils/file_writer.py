from pathlib import Path

def write_to_file(text: str, filepath: str):
    Path(filepath).open("a", encoding="utf-8").write(text)

def save_and_diff(original_text: str, fixed_text: str, doc_name: str, page_num: int = None):

    doc_name = ".".join(doc_name.split(".")[:-1])

    out_dir = Path("diff_review")
    out_dir.mkdir(exist_ok=True)

    original_path = out_dir / f"{doc_name}_original.txt"
    fixed_path = out_dir / f"{doc_name}_fixed.tex"

    page_splitter = "\n\n" + ("*"*180 + "\n") * 3 + "\n" + "\t" *3 + f"Page {page_num}" + "\n\n" if page_num is not None else ""
    segment_splitter = "\n\n" + "-"*180 + "\n\n"

    #original_text = original_text.replace("\n", "").replace("\r", "")
    #fixed_text = fixed_text.replace("\n", "").replace("\r", "")

    original_text = page_splitter + original_text + segment_splitter
    fixed_text = page_splitter + fixed_text + segment_splitter

    write_to_file(original_text, original_path)
    write_to_file(fixed_text, fixed_path)
    
    print(f"Saved original text to {original_path}")
    print(f"Saved fixed text to {fixed_path}")