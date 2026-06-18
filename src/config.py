import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(_PROJECT_ROOT, "data", "corpus")
CHROMA_DB_DIR = os.path.join(_PROJECT_ROOT, "data", "chroma_db")
COLLECTION_NAME = "hospital_docs"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
OLLAMA_MODEL = "llama3.2"