import sys
from vector_store import build_index
from rag import answer_question

def main():
    if "--rebuild" in sys.argv:
        n = build_index(reset=True)
        print(f"Indexed {n} chunks.\n")
    print("Type a question, or 'exit' to quit.\n")
    while True:
        q = input("Q: ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        r = answer_question(q)
        print("\nA:", r["answer"])
        print("Sources:", ", ".join(r["sources"]), "\n")

if __name__ == "__main__":
    main()