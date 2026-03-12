# Multilingual Semantic Document Assistant (Local RAG)

A high-performance, privacy-focused **Retrieval-Augmented Generation (RAG)** system that allows you to interact with your documents entirely locally. This assistant leverages state-of-the-art transformer models to provide context-aware answers from multiple PDF files without requiring any third-party APIs or cloud services.

## Key Features

**100% Local & Private:** No data leaves your machine. Perfect for processing sensitive or confidential documents.
**Multilingual Intelligence:** Seamlessly supports semantic search and question-answering across **English, Armenian, and Russian** using the `paraphrase-multilingual-MiniLM-L12-v2` embedding model.
**Scalable Vector Search:** Powered by **FAISS** (Facebook AI Similarity Search) for near-instant information retrieval across extensive document collections.
**Hardware Optimized:** Native support for **Apple Silicon (MPS)**, NVIDIA GPUs (CUDA), and multi-core CPUs for rapid local inference.
**User-Centric UI:** A clean, reactive web interface built with **Streamlit** for effortless document management and real-time chat.

## Technical Stack
| **Orchestration**| [LangChain](https://www.langchain.com/) (LCEL) |
| **LLM**          | [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| **Embeddings**   | [Sentence Transformers (Multilingual)](https://sbert.net/) |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) |
| **UI Framework** | [Streamlit](https://streamlit.io/) |

## Getting Started

### Prerequisites
* Python 3.9+
* Conda (Optional but recommended)

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone (https://github.com/MHakobyan04/multilingual-rag-assistant.git)
cd multilingual-rag-assistant
pip install -r requirements.txt
```
### 2. Launch the Application
Start the Streamlit server from your terminal:
```bash
streamlit run app.py
```

## How it Works

* **Ingestion**: Documents are loaded from the data/raw directory or uploaded directly through the web interface.
* **Chunking:**: The system utilizes a RecursiveCharacterTextSplitter to divide text into meaningful segments while maintaining contextual overlap.
* **Vectorization**: Each text chunk is transformed into a high-dimensional vector that captures its semantic essence.
* **Retrieval**: When a query is made, the system identifies the most relevant document segments using similarity search.
* **Generation**: The Local LLM synthesizes a final response based strictly on the retrieved context, significantly reducing the risk of hallucinations.

## Project Structure

multilingual-rag-assistant/
├── app.py                # Main Streamlit Web Interface
├── requirements.txt      # Project Dependencies
├── README.md             # Project Documentation
├── .gitignore            # Files to ignore in Git
├── src/
│   ├── document_loader.py # Multi-file PDF processing & chunking
│   ├── embeddings.py      # Multilingual vector model setup
│   └── rag_chain.py       # LLM initialization & RAG orchestration
└── data/
    ├── raw/              # Source PDF storage
    └── vectorstore/      # Local FAISS index persistence
