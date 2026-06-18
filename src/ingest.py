import os, glob
from config import CORPUS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_documents(corpus_dir=CORPUS_DIR):
    docs = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    if not docs:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")
    return docs

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks

def load_and_chunk_corpus(corpus_dir=CORPUS_DIR):
    records = []
    for doc in load_documents(corpus_dir):
        for i, chunk in enumerate(chunk_text(doc["text"])):
            records.append({"id": f"{doc['source']}::chunk{i}",
                             "source": doc["source"], "text": chunk})
    return records

if __name__ == "__main__":
    records = load_and_chunk_corpus()
    print(f"Loaded {len(records)} chunks.")