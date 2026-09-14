from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Directories

docs_dir = Path("./trimmed_docs")
persist_directory = "db/dbv1/chroma_db"



# Models

## Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True})

## Chat Model
chat_model = ChatOpenAI(model="openai.gpt-5.4-mini", temperature=0)