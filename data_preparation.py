import json

import numpy as np
from tqdm import tqdm
from text_splitter import RecursiveCharacterTextSplitter

from utils import get_pdf_files, get_pages_from_pdf

ts = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0, separators=["\n\n", "\n", ". ", " ", ""], keep_separator=False)

def get_pages_and_chunks_from_pdf(fp):
    for page_number, text in get_pages_from_pdf(fp):
        yield page_number, [s.replace('\n', ' ') for s in ts.split_text(text)]


def get_chunks(data_dir): 
    for fp in tqdm(list(get_pdf_files(data_dir))):
        url = 'https:/' + fp[len(data_dir):]
        for page_number, chunks in get_pages_and_chunks_from_pdf(fp):
            yield {"url": url + f'#page={page_number}', "chunks": chunks}

if __name__ == '__main__':
    data_dir = 'data'
    pages = list(get_chunks(data_dir))
    with open('data.json', 'w') as f:
        json.dump(pages, f, indent=2)