import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_directory(directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Loads all PDF documents from a specified directory and splits them
    into smaller, meaningful chunks for the vector database.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    print(f"Loading all documents from directory: {directory_path}...")
    # PyPDFDirectoryLoader automatically finds and reads all PDFs in the folder
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()

    if not documents:
        print("No PDF files found in the directory.")
        return []

    print("Starting text chunking process for all documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Successfully created {len(chunks)} text chunks from multiple files.")

    return chunks