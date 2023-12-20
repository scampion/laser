import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.11.0';

let response = await fetch("data.json");
const data = await response.json();

response = await fetch("augmented_data.json");
const augmented_data = await response.json();

let inputText = document.getElementById('inputText');
let outputDiv = document.getElementById('outputDiv');
let submitButton = document.getElementById('submitButton');

const status = document.getElementById('status');
status.textContent = '⏳ Loading model ..., please wait a few seconds';
const extractor = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5', { quantized: true });
const dim = 384;
status.textContent = 'Model loaded, loading vectors...';
let chunk_vectors = await loadBinaryFileFloat32('embeddings.chunks.bin', dim);
let title_vectors = await loadBinaryFileFloat32('embeddings.title.bin', dim);
let summary_vectors = await loadBinaryFileFloat32('embeddings.summary.bin', dim);

status.textContent = 'Vectors loaded';
status.textContent = 'Ready'; 
status.textContent = ''; 


async function loadBinaryFileFloat32(filePath, vectorSize) {
    try {
        // Read the binary file synchronously.
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const binaryData = await response.arrayBuffer();
        const vectors = [];
        // load float64 vectors
        for (let i = 0; i < binaryData.byteLength; i += vectorSize * 4) {
            const vector = binaryData.slice(i, i + vectorSize * 4);
            vectors.push(new Float32Array(vector));
        }
        return vectors;
    } catch (error) {
        console.error("Error loading binary file: ", error);
        return null;
    }
}

function dotProduct(vecA, vecB){
    let product = 0;
    for(let i=0;i<vecA.length;i++){
        product += vecA[i] * vecB[i];
    }
    return product;
}

function magnitude(vec){
    let sum = 0;
    for (let i = 0;i<vec.length;i++){
        sum += vec[i] * vec[i];
    }
    return Math.sqrt(sum);
}

function cosineSimilarity(vecA,vecB){
    return dotProduct(vecA,vecB)/ (magnitude(vecA) * magnitude(vecB));
}

function get_chunk_and_url(n){
    // parse data and count numbers of chunks when n chunks are found return the chunk
    let chunk = "";
    let chunk_count = 0;
    for(let i = 0; i < data.length; i++){
        let url = data[i]['url'];
        let chunks = data[i]['chunks'];
        for(let j = 0; j < chunks.length; j++){
            chunk_count += 1;
            chunk = chunks[j];
            if(chunk_count == n){
                return [url, chunk];
            }
        }
    }
    console.log("ERROR: chunk not found " + n + " " + chunk_count)  ;
}

async function similarities(query, vectors){
    let result = vectors.map((vector, index) => {
        return {
            label: index,
            value: cosineSimilarity(query.data, vector)
        };
    }).sort((a, b) => b.value - a.value);
    return result;
}

async function search(question){
    let query = await extractor(question, { pooling: 'mean', normalize: true });
    let chunk_results = await similarities(query, chunk_vectors);
    let title_results = await similarities(query, title_vectors);
    let summary_results = await similarities(query, summary_vectors);
    let [sorted_score_per_url, grouped_by_url] = rerank(chunk_results, title_results, summary_results);
    display_results(sorted_score_per_url, grouped_by_url);
}

function get_title(url){
    for(let i = 0; i < augmented_data.length; i++){
        if(augmented_data[i]['url'] == url){
            return augmented_data[i]['title'];
        }
    }
}

function rerank(chunk_results, title_results, summary_results){
    let score_per_document_by_title = {};
    let score_per_document_by_summary = {};

    for(let i = 0; i < augmented_data.length; i++){
        let url = augmented_data[i]['url'];
        score_per_document_by_title[url] = title_results[i].value;
        score_per_document_by_summary[url] = summary_results[i].value;
    }

    let score_per_chunk = {};
    for(let i = 0; i < chunk_results.length; i++){
        let chunk_result = chunk_results[i];
        let url = get_chunk_and_url(chunk_result.label + 1)[0];
        url = url.split("#")[0];
        
        score_per_chunk[parseInt(chunk_result.label) + 1 ] = chunk_result.value * 3 + score_per_document_by_title[url] * 1.5 + score_per_document_by_summary[url];
    }

    // sort by score
    let sorted_score_per_chunk = Object.keys(score_per_chunk).sort(function(a,b){return score_per_chunk[b]-score_per_chunk[a]});

    let top_100_chunks = sorted_score_per_chunk.slice(0, 100);

    // group by url and get the top 10 chunks per url and mean score
    let grouped_by_url = {};
    let score_per_url = {};
    for(let i = 0; i < top_100_chunks.length; i++){
        let chunk_id = parseInt(top_100_chunks[i]);
        let url = get_chunk_and_url(chunk_id + 1)[0];
        url = url.split("#")[0];
        if(!(url in grouped_by_url)){
            grouped_by_url[url] = [];
            score_per_url[url] = 0;
        }
        grouped_by_url[url].push(chunk_id);
        score_per_url[url] += score_per_chunk[chunk_id];
    }
    // mean score per url
    for(let url in score_per_url){
        score_per_url[url] /= grouped_by_url[url].length;
    }
    // sort by score
    let sorted_score_per_url = Object.keys(score_per_url).sort(function(a,b){return score_per_url[b]-score_per_url[a]});

    return [sorted_score_per_url, grouped_by_url];
}

function display_results(sorted_score_per_url, grouped_by_url){
    outputDiv.innerHTML = "";
    let ul = document.createElement('ul'); 
    for(let i = 0; i < sorted_score_per_url.length; i++){
        let url = sorted_score_per_url[i];
        let chunks = grouped_by_url[url];
        var li = document.createElement('li');
        // create title with h3 tag
        var h3 = document.createElement('h3');
        var a = document.createElement('a');
        a.setAttribute('href', url);
        a.setAttribute('target', '_blank');
        a.appendChild(document.createTextNode("🔗 " +get_title(url)));
        h3.appendChild(a);
        li.appendChild(h3);

        for(let j = 0; j < Math.min(5, chunks.length); j++){
            let chunk_id = chunks[j];
            let [url, chunk] = get_chunk_and_url(chunk_id + 1);
            var span = document.createElement('span');
            span.setAttribute('class', 'citation');
            span.appendChild(document.createTextNode(chunk));
            span.appendChild(document.createElement('br'));
            span.appendChild(document.createElement('br'));
            var a = document.createElement('a');
            a.setAttribute('href', url);
            a.setAttribute('target', '_blank');
            a.appendChild(document.createTextNode(url));
            span.appendChild(a);
            li.appendChild(span);   
            li.appendChild(document.createElement('hr'));
        }
        ul.appendChild(li);        
    }   
    outputDiv.appendChild(ul);
}

async function search_legacy(question){
    let query = await extractor(question, { pooling: 'mean', normalize: true });
    let result = vectors.map((vector, index) => {
        return {
            label: index,
            value: cosineSimilarity(query.data, vector)
        };
    }).sort((a, b) => b.value - a.value).slice(0, 10);
    outputDiv.innerHTML = "";
    let ul = document.createElement('ul');    
    for(let i = 0; i < result.length; i++){
        let [url, chunk] = get_chunk_and_url(result[i].label + 1);
        var li = document.createElement('li');
        li.appendChild(document.createTextNode("⭐️".repeat(Math.ceil(result[i].value * 5 ) + " ")));
        li.appendChild(document.createElement('br'));
        li.appendChild(document.createTextNode(chunk));
        li.appendChild(document.createElement('br'));
        
        var a = document.createElement('a');
        a.setAttribute('href', url);
        a.setAttribute('target', '_blank');
        
        a.appendChild(document.createTextNode("📄 PDF link "));
        li.appendChild(document.createElement('br'));
        li.appendChild(a);
        ul.appendChild(li);        
    }
    outputDiv.appendChild(ul);
}

submitButton.addEventListener('click', async (e) => {
    e.preventDefault();
    search(inputText.value);
});

inputText.addEventListener('keyup', async (e) => {
    e.preventDefault();
    search(inputText.value);
});
