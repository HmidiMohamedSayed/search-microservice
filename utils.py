import numpy as np
from scipy import spatial
from typing import List
from sentence_transformers import SentenceTransformer
import json
from bson.objectid import ObjectId
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import os
import pinecone
import re

def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances

def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    """Return a list of indices of nearest neighbors from a list of distances."""
    return np.argsort(distances)

def embed(transcript_srt):
    """
    Embeds transcript for search
    
    Arguments: 
        transcript_srt: transcript string in srt format
    Returns:
        index (Pinecone index): index containing all embedding vectors, their ids + metadata
        model (Sentence Transformer): Model used to produce embeddings
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    #put transcript srt text into srt file
    file_obj = open("indexation.srt", "w")
    file_obj.write(transcript_srt)
    
    #-------------srt to json------------------
    regex = r'(?:\d+)\s(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\s+(.+?)(?:\n\n|$)'
    offset_seconds = lambda ts: sum(howmany * sec for howmany, sec in zip(map(int, ts.replace(',', ':').split(':')), [60 * 60, 60, 1, 1e-3]))
    transcript = [dict(startTime = offset_seconds(startTime), endTime = offset_seconds(endTime), ref = ' '.join(ref.split())) for startTime, endTime, ref in re.findall(regex, open('indexation.srt').read(), re.DOTALL)]
    with open('data.json','w', encoding='utf-8') as f:
        json.dump(transcript ,f)
        data = []
        data = transcript
    #-----------create new data clusters + overlap----------
    new_data = []
    window = 6 # number of sentences to comine
    stride = 3 # numer of sentences to 'stride' over, used to create overlap

    for i in tqdm(range(0, len(data), stride)):
        i_end = min(len(data)-1, i+window)
        text = ' '
        for j in range(i,i_end+1):
            text = text + data[j]['ref']
            j+1
        new_data.append({
            'startTime': data[i]['startTime'],
            'endTime': data[i_end]['endTime'],
            'ref': text
        })

    #--------encode data with a QA embedding model------------
    model_id = "multi-qa-mpnet-base-dot-v1"
    model = SentenceTransformer(model_id)
    dim = model.get_sentence_embedding_dimension()
    pinecone.init(
        pinecone_api_key= pinecone_api_key,  # app.pinecone.io
        environment="asia-southeast1-gcp-free"  # find next to API key
    )
    index_id = "smart"
    if index_id not in pinecone.list_indexes():
        pinecone.create_index(
            index_id,
            dim,
            metric="dotproduct"
        )
    index = pinecone.Index(index_id)

    #-----------inserting the embeddings (and metadata) into our index--------
    batch_size = 32
    insertList=[]
    # loop through the batches
    for i in tqdm(range(0, len(new_data), batch_size)):
        # find end position of batch (for when we hit end of data)
        i_end = min(len(new_data)-1, i+batch_size)
        # extract the metadata like text, start/end positions, etc
        batch_meta = [{
            "ref": new_data[x]["ref"],
            "startTime": new_data[x]["startTime"],
            "endTime": new_data[x]["endTime"],
        } for x in range(i, i_end)]
        batch_text = [
            row['ref'] for row in new_data[i:i_end]
        ]
        batch_embeds = model.encode(batch_text).tolist()
        ids_batch = [str(n) for n in range(i, i_end)]
        to_upsert = list(zip(ids_batch, batch_embeds, batch_meta))
        index.upsert(to_upsert)
        insertList.append(to_upsert)
    return index, model