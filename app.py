import streamlit as st
import os
from src.document_loader import load_and_chunk_directory
from src.embeddings import get_embedding_model, create_and_save_vectorstore, load_vectorstore
from src.rag_chain import setup_llm, create_rag_chain

# Configuration
RAW_DATA_DIR = "data/raw"
VECTORSTORE_PATH = "data/vectorstore"

# Page config
st.set_page_config(page_title="AI Document Assistant", page_icon="📚")
st.title("AI Document Assistant")
st.markdown("---")

# Session state initialization to keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for configuration and file uploads
with st.sidebar:
    st.header("Settings")
    uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=['pdf'])

    if st.button("Build/Update Knowledge Base"):
        if uploaded_files:
            # Save uploaded files to data/raw
            if not os.path.exists(RAW_DATA_DIR):
                os.makedirs(RAW_DATA_DIR)

            for uploaded_file in uploaded_files:
                with open(os.path.join(RAW_DATA_DIR, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

            st.info("Processing documents... Please wait.")
            # Clear old vectorstore if it exists
            if os.path.exists(VECTORSTORE_PATH):
                import shutil

                shutil.rmtree(VECTORSTORE_PATH)

            # Process and build
            embedding_model = get_embedding_model()
            chunks = load_and_chunk_directory(RAW_DATA_DIR)
            create_and_save_vectorstore(chunks, embedding_model, VECTORSTORE_PATH)
            st.success("Knowledge base ready!")
        else:
            st.warning("Please upload some PDFs first.")

# Main Chat Interface

# Load core components only once
if "rag_chain" not in st.session_state:
    if os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
        emb_model = get_embedding_model()
        v_store = load_vectorstore(emb_model, VECTORSTORE_PATH)
        llm_model = setup_llm()
        st.session_state.rag_chain = create_rag_chain(v_store, llm_model)
    else:
        st.info("Please upload documents and build the knowledge base to start.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if "rag_chain" in st.session_state:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                response = st.session_state.rag_chain.invoke(prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("RAG chain not initialized. Please build the knowledge base.")