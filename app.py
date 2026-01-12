import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEmbeddings,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ENV
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN not found")

# STREAMLIT
st.set_page_config(page_title="Multi-PDF Chat (LangChain v1.0)", layout="wide")
st.title("📄 Chat with Multiple PDFs")

# LLM
llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.5,
    huggingfacehub_api_token=HF_TOKEN,
    provider="auto",
)
llm = ChatHuggingFace(llm=llm_endpoint)

# PDF PROCESSING
def process_pdfs(uploaded_files):
    documents = []

    for pdf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents.extend(loader.load())
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# SIDEBAR
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("🚀 Process PDFs"):
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                st.session_state.vectorstore = process_pdfs(uploaded_files)
            st.success("PDFs ready!")
        else:
            st.warning("Upload at least one PDF")

# RAG PROMPT
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
Use ONLY the context below to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}
"""
)

# CHAT
if "vectorstore" in st.session_state:
    st.subheader("💬 Ask a Question")

    question = st.text_input("Ask something about the PDFs")

    if question:
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(question)

        st.markdown("### 🤖 Answer")
        st.write(answer)

else:
    st.info("👈 Upload PDFs and process them to start chatting.")
