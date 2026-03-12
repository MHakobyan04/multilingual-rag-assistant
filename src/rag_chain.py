import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


def setup_llm(model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Initializes the LLM with universal device support (MPS for Mac, CUDA for NVIDIA, or CPU).
    """
    print(f"Loading Local LLM: {model_id}...")

    # Logic to select the best available hardware accelerator
    if torch.backends.mps.is_available():
        device = "mps"
        print("Acceleration: Apple Silicon MPS detected.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Acceleration: NVIDIA CUDA detected.")
    else:
        device = "cpu"
        print("Running on standard CPU.")

    hf_pipeline = pipeline(
        "text-generation",
        model=model_id,
        device=device,
        max_new_tokens=256,
        return_full_text=False,
        do_sample=False
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm

def create_rag_chain(vectorstore, llm):
    """
    Builds the Retrieval-Augmented Generation (RAG) pipeline using LCEL
    (LangChain Expression Language).
    """
    # Convert the FAISS database into a retriever object
    # k=3 means it will fetch the top 3 most relevant text chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    # Define the template that instructs the LLM on how to answer
    template = """Use the following pieces of retrieved context to answer the question.
    If you don't know the answer based on the context, just say that you don't know.
    Keep the answer concise and strictly based on the provided text.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    prompt = PromptTemplate.from_template(template)
    # Function to format the retrieved documents into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    print("Constructing the RAG chain...")

    # Building the LCEL chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain is ready for queries.")

    return rag_chain
