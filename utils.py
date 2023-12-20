import os
from pypdf import PdfReader


def get_pdf_files(data_dir):
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('en.pdf'):
                yield os.path.join(root, file)

def get_pages_from_pdf(fp):
    pdfReader = PdfReader(fp)
    for page_number, page in enumerate(pdfReader.pages):
        yield page_number, page.extract_text()

def get_text_from_pdf(fp):
    text = ''
    for _, page in enumerate(PdfReader(fp).pages):
        text += page.extract_text()
    return text
    
