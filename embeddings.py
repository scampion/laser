import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def write_embeddings(data, key):
    with open(f"embeddings.{key}.bin", 'wb') as out:
        for d in tqdm(data):
            embeddings = model.encode(d[key])
            out.write(embeddings.tobytes())

if __name__ == '__main__':
    data_dir = 'data'
    with open('data.json') as f:
        pages = json.load(f)
        write_embeddings(pages, 'chunks')
    
    # Compute augmented embeddings
    with open('augmented_data.json') as f:
        augmented_data = json.load(f)
        write_embeddings(augmented_data, 'title')
        write_embeddings(augmented_data, 'summary')


            
