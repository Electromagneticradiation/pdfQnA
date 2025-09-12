import os
import streamlit as st
from pathlib import Path
import pdfplumber
from groq import Groq
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from pdf2image import convert_from_path
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def extract_text_chunks(pdf_path, chunk_size=500, overlap=50):
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                all_text.append(text)
            else:
                # OCR fallback
                images = convert_from_path(pdf_path, first_page=page.page_number, last_page=page.page_number)
                if images:
                    ocr_text = pytesseract.image_to_string(images[0])
                    if ocr_text.strip():
                        all_text.append(ocr_text)

    # Join all text into one string
    full_text = "\n".join(all_text)

    # Chunk into overlapping windows
    return chunk_text(full_text, chunk_size, overlap)


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]

# --- Deduplicate ---
def clean_chunks(chunks):
    seen = set()
    cleaned = []
    for c in chunks:
        if c not in seen:
            cleaned.append(c)
            seen.add(c)
    return cleaned


def build_chroma(pdf_path, persist_dir="chroma_store"):
    # 1. Extract full text
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if not text.strip():
                # OCR fallback
                images = convert_from_path(pdf_path, first_page=page.page_number, last_page=page.page_number)
                if images:
                    text = pytesseract.image_to_string(images[0])
            if text.strip():
                all_text.append(text)

    # 2. Join all pages into a single string
    full_text = "\n".join(all_text)

    # 3. Chunk into overlapping segments
    chunks = chunk_text(full_text, chunk_size=800, overlap=100)

    # 4. Store in Chroma
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
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

st.set_page_config(page_title="PDF QnA", page_icon="📄", layout="centered")

st.title("📄 PDF Question & Answer")
st.write("Upload a PDF and ask questions — powered by HuggingFace embeddings + Groq LLM.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    tmp_path = Path("uploaded.pdf")
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("🔍 Processing PDF (extracting text + OCR fallback)..."):
        vectordb = build_chroma(tmp_path)

    st.success("✅ PDF processed! You can now ask questions.")

    st.markdown("---")
    st.subheader("❓ Ask a question about your PDF")

    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("🤖 Thinking..."):
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)

            context = "\n\n".join(dict.fromkeys([d.page_content for d in docs]))
            answer = query_groq(query, context)

        st.markdown("### 📝 Answer")
        st.write(answer)

        with st.expander("📑 Show retrieved context"):
            st.text(context)
else:
    st.info("⬆️ Please upload a PDF to get started.")
