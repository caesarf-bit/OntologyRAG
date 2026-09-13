from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

docs_dir = Path("./trimmed_docs")
persist_directory = "dbv1/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True})