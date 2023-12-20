all: data data.json augmented_data.json embeddings.chunks.bin embeddings.title.bin embeddings.summary.bin

data:
	mkdir -p data
	cd data && wget -r -l 1 -nc -A pdf https://www.europarl.europa.eu/compendium/en/contents


data.json:
	python data_preparation.py

augmented_data.json:
	python augment.py


embeddings.chunks.bin embeddings.title.bin embeddings.summary.bin:
	python embeddings.py
	

clean:
	rm -rf data data.json augmented_data.json embeddings.chunks.bin embeddings.title.bin embeddings.summary.bin
