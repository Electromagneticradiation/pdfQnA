import pdfplumber                                                       # for textual & tabular content
from pdf2image import convert_from_path
import pytesseract                                                      # for image-text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

''' function to extract content from the uploaded PDF
by default : extracts text & tables. if neither is found, extracts text from images using OCR
if content_type is specified then extracts only those types '''

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

''' function to create and persist Chroma vector index from the chunks '''

def create_chroma_index(chunks, persist_directory="chroma_db"):
    # Use HuggingFace sentence-transformers model for embeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Store in Chroma
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb


