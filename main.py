import pdfplumber                             # for textual & tabular content
from pdf2image import convert_from_path
import pytesseract                            # for image-text extraction

"""
function to extract content from the uploaded PDF
by default : extracts text & tables. if neither is found, extracts text from images using OCR
if content_type is specified then extracts only those types 
"""

def content_extractor(pdf_path, content_type = ['text', 'tables']):
    
    content = {"text": [], "tables": [], "ocr text": []}

    if ('text' in content_type) or ('tables' in content_type) :
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:

                # text extraction
                if 'text' in content_type:
                    text = page.extract_text()
                    if text:
                        content["text"].append(text)
                
                # table extraction
                if 'tables' in content_type:
                    tables = page.extract_tables()
                    if tables:
                        content["tables"].extend(tables)
    
    # either the user selected textual images OR no text/tables were found
    if ('textual images' in content_type) ^ (not(content["text"] or content["tables"])) :
        page_image = convert_from_path(pdf_path)
        for img in page_image :
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip() :
                content["ocr text"].append(ocr_text)
    
    return content
