import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import ollama
from config import CHROMA_DB_DIR, COLLECTION_NAME, TOP_K
from ingest import load_and_chunk_corpus

EMBED_MODEL = "nomic-embed-text"

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in input]

def get_client():
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)

def get_collection(client=None, reset=False):
    client = client or get_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )

def build_index(reset=True):
    records = load_and_chunk_corpus()
    collection = get_collection(reset=reset)
    collection.add(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[{"source": r["source"]} for r in records],
    )
    return len(records)

def retrieve(query, top_k=TOP_K):
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    return [{"text": t, "source": m["source"], "distance": d}
            for t, m, d in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])]

if __name__ == "__main__":
    n = build_index(reset=True)
    print(f"Indexed {n} chunks.")
    for hit in retrieve("test query"):
        print(hit["source"], hit["distance"])