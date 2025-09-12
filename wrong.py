import io
import os
import base64
import requests
import pdfplumber
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from pdf2image import convert_from_path
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# we'll use jina for multimodal embedding model & groq for llm
load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {JINA_API_KEY}"}

# Option 1: Text extraction
def extract_text_chunks(pdf_path):
    content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                content.append(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text("\n\n".join(content))

def build_chroma_from_text(chunks, persist_dir="chroma_text"):
    encoder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = [Document(page_content=chunk, metadata={"source": i}) for i, chunk in enumerate(chunks)]
    vectordb = Chroma.from_documents(docs, embedding=encoder, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

# Option 2: Image → Jina multimodal

class PrecomputedEmbeddings(Embeddings):
    def __init__(self, vectors):
        self.vectors = vectors

    def embed_documents(self, texts):
        if len(texts) != len(self.vectors):
            raise ValueError("Number of texts and vectors must match")
        return self.vectors

    def embed_query(self, text):
        # Not used for batch, but needed for Chroma
        return [0.0] * len(self.vectors[0])
    

def pdf_to_img(pdf_path):
    return convert_from_path(pdf_path, dpi=200)

def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_jina_embeddings(items, task = "retrieval.passage"):
    payload = {
        "model": "jina-embeddings-v4",
        "task": task,
        "input": items
    }
    resp = requests.post(JINA_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


def build_chroma_from_jina(pdf_path, persist_dir="chroma_jina"):
    images = pdf_to_img(pdf_path)
    encoded = [encode_image(img) for img in images]
    embeddings = get_jina_embeddings([{"image": img_b64} for img_b64 in encoded])
    docs = [Document(page_content=f"Page {i+1}", metadata={"page": i+1}) for i in range(len(images))]

    # wrap embeddings in dummy object to satisfy Chroma
    class PrecomputedEmbeddings:
        def embed_documents(self, texts):
            return embeddings
        def embed_query(self, text):
            return get_jina_embeddings([{"text": text}], task="retrieval.query")[0]

    vectordb = Chroma.from_texts(
        texts=[d.page_content for d in docs],
        embedding=PrecomputedEmbeddings(),
        metadatas=[d.metadata for d in docs],
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

# Query using Groq
def query_groq(question, context):
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are the most badass assistant. Now answer the questions from the document like a pro."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return completion.choices[0].message.content

# Query using Jina (for multimodal)
def query_jina(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)
    return "\n\n".join([d.page_content for d in docs])

# -------------------
# Streamlit UI
# -------------------
st.title("📄 PDF Q&A Assistant")

mode = st.radio("Choose mode:", ["Option 1: Text-based (English PDFs)", "Option 2: Multilingual/Complex (Jina multimodal)"])

uploaded = st.file_uploader("Upload PDF", type="pdf")
if uploaded:
    tmp_path = f"tmp_{uploaded.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    if mode.startswith("Option 1"):
        st.info("Using pdfplumber + HuggingFace embeddings")
        chunks = extract_text_chunks(tmp_path)
        vectordb = build_chroma_from_text(chunks)
    else:
        st.info("Using Jina multimodal embeddings")
        vectordb = build_chroma_from_jina(tmp_path)

    query = st.text_input("Ask a question about the PDF:")
    if query:
        if mode.startswith("Option 1"):
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([d.page_content for d in docs])
        else :
            context = query_jina(query, vectordb)

        answer = query_groq(query, context)
        st.subheader("Answer:")
        st.write(answer)

        if st.button("Show retrieved context"):
            st.text(context)
