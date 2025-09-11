import os
import json
import base64
import requests
from dotenv import load_dotenv
from pdf2image import convert_from_path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

### jina API setup cz we'll use its multimodal embedding model ###
load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {JINA_API_KEY}"
}

#1 convert PDF pages into images
def pdf_to_img(pdf_path):
    return convert_from_path(pdf_path, dpi = 200)

#2 encode image to base64 string
def encode_image(img):
    import io
    buf = io.BytesIO()
    img.save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

#3 get embeddings from jina
def get_embeddings(images):
    payload = {
        "model": "jina-embeddings-v4",
        "task": "retrieval.passage",
        "input": [{"image": img_b64} for img_b64 in images]
    }
    resp = requests.post(JINA_URL, headers = HEADERS, json = payload)
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]

#4 store in chroma
def store_in_chroma(embeddings, metadatas):
    texts = [f"Page {i}" for i in range(len(embeddings))]  # dummy text labels
    vectordb = Chroma.from_embeddings(
        embeddings=embeddings,
        texts=texts,
        metadatas=metadatas,
        persist_directory = "chroma_db"
    )
    vectordb.persist()
    return vectordb

#5 query chroma db
def query_chroma(vectordb, query_text, top_k = 3):
    results = vectordb.similarity_search(query_text, k = top_k)
    for res in results:
        print(f"Page {res.metadata['page']} | Score: {res.score}")
        print(f"Text Label: {res.page_content}\n")
    return results

### MAIN PIPELINE ###
def process_pdf(pdf_path):
    print(f"Converting {pdf_path} to images...")
    images = pdf_to_img(pdf_path)

    print("Encoding images...")
    encoded = [encode_image(img) for img in images]

    print("Getting embeddings from Jina...")
    embeddings = get_embeddings(encoded)

    print("Storing in Chroma...")
    metadatas = [{"page": i+1} for i in range(len(images))]
    vectordb = store_in_chroma(embeddings, metadatas)

    print("Done.")
    return vectordb
