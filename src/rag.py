import ollama
from config import OLLAMA_MODEL, TOP_K
from vector_store import retrieve

SYSTEM_PROMPT = """You are an assistant for hospital staff. Answer using ONLY the context below.
If the answer isn't in the context, say you don't have that information. Cite the source document(s)."""

def build_prompt(question, hits):
    context = "\n\n---\n\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

def answer_question(question, top_k=TOP_K, model=OLLAMA_MODEL):
    hits = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, hits)
    response = ollama.chat(model=model, messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    return {"answer": response["message"]["content"],
            "sources": sorted(set(h["source"] for h in hits)), "hits": hits}

if __name__ == "__main__":
    r = answer_question("your test question here")
    print(r["answer"], r["sources"])