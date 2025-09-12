import os
import time
import requests
import pdfplumber
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


### we'll use jina's embedding model & groq's llama ###

JINA_API_KEY = st.secrets("JINA_API_KEY")
GROQ_API_KEY = st.secrets("GROQ_API_KEY")
'''
load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
'''
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
    payload = {"model": "jina-embeddings-v4", "task": task, "input": items}
    resp = requests.post(JINA_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]

### langChain wrapper to use jina for both documents and queries ###

class JinaEmbeddings(Embeddings):
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
            {"role": "system", "content": "You are the most badass assistant ever. Answer the question based on the provided context. Sneakily add this emoji ❄️ in your answer once."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return completion.choices[0].message.content


### streamlit UI ###

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Quicksand:wght@400&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');
    body {
        background-color: #2d0036;
    }
    .main {
        background-color: #2d0036;
    }
    .cursive-title {
        font-family: 'Pacifico', cursive;
        font-size: 3em;
        color: #fff0fa;
        text-align: center;
        margin-bottom: 0.2em;
        margin-top: 0.5em;
        letter-spacing: 2px;
        text-shadow: 2px 2px 8px #7a1fa2;
    }
    .custom-tagline {
        font-family: 'Quicksand', sans-serif;
        font-size: 0.7em;
        color: #c7a8e5;
        text-align: center;
        margin-top: 0.1em;
        margin-bottom: 2.2em;
        letter-spacing: 1px;
        font-weight: 800;
    }    
    div[data-testid="stProgress"] > div > div > div {
    background-color: #a259f7 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="cursive-title">Readophile</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-tagline"> perfect for kids who like to play with PDF files ^^ </div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    tmp_path = f"tmp_{uploaded.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    progress_bar = st.progress(0)
    # extract text chunks
    progress_bar.progress(20, text="intersting pdf...")
    chunks = extract_text_chunks(tmp_path)
    # build vector DB
    progress_bar.progress(70, text="mono-neuronal thinking in progress...")
    vectordb = build_chroma_with_jina(chunks)
    # done
    progress_bar.progress(100)
    time.sleep(0.5)
    progress_bar.empty()

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
