import os
import requests
import pdfplumber
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# we'll use jina for embedding model & groq for llm

load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {JINA_API_KEY}"}


def extract_text_chunks(pdf_path):
    content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                content.append(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text("\n\n".join(content))

def get_jina_embeddings(items, task="retrieval.passage"):
    """Send text(s) to Jina for embeddings."""
    payload = {"model": "jina-embeddings-v4", "task": task, "input": items}
    resp = requests.post(JINA_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]

class JinaEmbeddings(Embeddings):
    """LangChain wrapper to use Jina for both documents and queries."""

    def embed_documents(self, texts):
        inputs = [{"text": t} for t in texts]
        return get_jina_embeddings(inputs, task="retrieval.passage")

    def embed_query(self, text):
        return get_jina_embeddings([{"text": text}], task="retrieval.query")[0]

def build_chroma_with_jina(chunks, persist_dir="chroma_db"):
    """Build vector DB from chunks using Jina embeddings."""
    docs = [Document(page_content=chunk, metadata={"chunk": i}) for i, chunk in enumerate(chunks)]
    vectordb = Chroma.from_documents(docs, embedding=JinaEmbeddings(), persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def query_groq(question, context):
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are the most badass assistant. Now answer the questions based on given context like a pro"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return completion.choices[0].message.content

# -------------------
# Streamlit UI
# -------------------
st.title("📄 PDF Q&A with Jina + Groq")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    tmp_path = f"tmp_{uploaded.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info("Extracting & embedding with Jina...")
    chunks = extract_text_chunks(tmp_path)
    vectordb = build_chroma_with_jina(chunks)

    query = st.text_input("Ask a question about this PDF:")
    if query:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])

        answer = query_groq(query, context)

        st.subheader("Answer:")
        st.write(answer)

        if st.button("Show retrieved context"):
            st.text(context)
