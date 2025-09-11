import pdfplumber                                                       # for textual & tabular content
from pdf2image import convert_from_path
import pytesseract                                                      # for image-text extraction
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import Embeddings
from dotenv import load_dotenv
import os

load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")

def content_extractor(pdf_path, content_type = ['text', 'tables']):
    
    content = []

    if ('text' in content_type) or ('tables' in content_type) :
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:

                # text extraction
                if 'text' in content_type:
                    text = page.extract_text()
                    if text:
                        content.append(text)
                
                # table extraction
                if 'tables' in content_type:
                    tables = page.extract_tables()
                    for table in tables:
                        table_str = "\n".join(["\t".join(row) for row in table if row])
                        content.append(table_str)

    # either the user selected textual images OR no text/tables were found
    if ('textual images' in content_type) ^ (not content) :
        page_image = convert_from_path(pdf_path)
        for img in page_image :
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip() :
                content.append(ocr_text)
    
    return "\n\n".join(content) 

''' function to chunk the extracted content into smaller pieces '''

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

''' function to create and persist chroma vector index from the chunks '''


# ---------------------------
# Custom Jina Embeddings Wrapper
# ---------------------------
class JinaEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "jina-embeddings-v4"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            payload = {
                "model": self.model,
                "task": "text-matching",
                "input": [{"text": text}]
            }
            response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["data"][0]["embedding"])
        return embeddings

    def embed_query(self, text):
        payload = {
            "model": self.model,
            "task": "text-matching",
            "input": [{"text": text}]
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


# ---------------------------
# PDF Content Extractor
# ---------------------------
def content_extractor(pdf_path, content_type=['text', 'tables']):
    content = []

    if ('text' in content_type) or ('tables' in content_type):
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:

                # text extraction
                if 'text' in content_type:
                    text = page.extract_text()
                    if text:
                        content.append(text)

                # table extraction
                if 'tables' in content_type:
                    tables = page.extract_tables()
                    for table in tables:
                        table_str = "\n".join(["\t".join(row) for row in table if row])
                        content.append(table_str)

    # OCR fallback
    if ('textual images' in content_type) ^ (not content):
        page_images = convert_from_path(pdf_path)
        for img in page_images:
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                content.append(ocr_text)

    return "\n\n".join(content)


# ---------------------------
# Text Chunking
# ---------------------------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


# ---------------------------
# Chroma Index with Jina Embeddings
# ---------------------------
def chroma_index(chunks, api_key = JINA_API_KEY):
    encoder = JinaEmbeddings(api_key = api_key)
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=encoder,
        persist_directory = "chroma_db"
    )
    vectordb.persist()
    return vectordb

