import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Switching to a multilingual model to support Russian, Armenian, and other languages
# This model maps sentences from 50+ languages into the same vector space
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """
    Initializes and returns the Hugging Face embedding model.
    Downloads the model to the local cache on the first run.
    """
    print(f"Loading embedding model: {model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def create_and_save_vectorstore(chunks, embedding_model, save_path: str = "data/vectorstore"):
    """
    Converts text chunks into numerical vectors, stores them in a FAISS database,
    and saves the database to the local file system.
    """
    print(f"Creating FAISS vector database from {len(chunks)} text chunks")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    print(f"Saving vector database to: {save_path}")
    vectorstore.save_local(save_path)
    print(f"Vector database succesfully saved.")

    return vectorstore

def load_vectorstore(embedding_model, load_path: str = "data/vectorstore"):
    """
    Loads an existing FAISS vector database from the local directory for quick querying.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Vector database not found at {load_path}. Please build it first.")

    print(f"Loading existing vector database from: {load_path}.")
    # allow_dangerous_deserialization is required in newer LangChain versions for local FAISS files
    vectorstore = FAISS.load_local(
        load_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    print(f"Vector database succesfully loaded.")

    return vectorstore
