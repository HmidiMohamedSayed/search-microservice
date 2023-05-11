from flask import Flask, jsonify, request, render_template
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import openai
from bson.objectid import ObjectId
from sentence_transformers import SentenceTransformer
import pinecone
from utils import distances_from_embeddings, indices_of_nearest_neighbors_from_distances



app = Flask(__name__, template_folder='static')
load_dotenv()
#connect to db and get collections
db_uri = os.getenv("MONGO_HOST")
db_name = os.getenv("DB_NAME")
db_port = os.getenv("MONGO_PORT")
client = MongoClient(host=db_uri, port=int(db_port), server_api=ServerApi('1'))
api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai.api_key = api_key  
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
        print(e)

db = client[db_name]
synthesis_collection = db["synthesis"]
transcript_collection = db["transcript"]



@app.route(f"/search", methods=["POST"])
def search():
    key = request.args.get('key')    
    print(key)
    true_var=True
    syntheses = synthesis_collection.find({"isPublic" : true_var}, {"_id": 1, "embedding": 1})
    syntheses = list(syntheses)
    key_embedding = openai.Embedding.create(input = key, model="text-embedding-ada-002")['data'][0]['embedding']
    indices = []
    ids_dic = dict()
    for index in range(len(syntheses)):
        ids_dic[index] = syntheses[index]["_id"]
    embeddings = [synthesis["embedding"] for synthesis in syntheses]
    distances = distances_from_embeddings(key_embedding, embeddings)
    distances = [distance for distance in distances if distance < 0.2]
    indices = indices_of_nearest_neighbors_from_distances(distances)
    _ids = [ids_dic[index] for index in indices]
    # retrieve synthesis names with indices
    search_result = synthesis_collection.find({"_id": {"$in": _ids}}, {"_id": 1, "title": 1 , "text" :1 , "tags":1 })
    search_result = list(search_result)    
    results=[]
    for search in search_result :
        results.append({'_id':str(search["_id"]),'title':search["title"],'text':search["text"],'tags':search["tags"]})
    print(results)
    print(search_result)
    # Return synthesized document
    return results

@app.route(f"/searchConcept", methods=["POST"])
def searchConcept():
    query = request.args.get('query')
    transcript_id = request.args.get('transcript_id')    
    print(query)
    print(transcript_id)
    #get model
    model_id = "multi-qa-mpnet-base-dot-v1"
    model = SentenceTransformer(model_id)
    dim = model.get_sentence_embedding_dimension()
    pinecone.init(
            pinecone_api_key= pinecone_api_key,  # app.pinecone.io
            environment="asia-southeast1-gcp-free"  # find next to API key
        )
     #create index
    index_id = "smart"
    if index_id not in pinecone.list_indexes():
            pinecone.create_index(
                index_id,
                dim,
                metric="dotproduct"
            )
    index = pinecone.Index(index_id)
    print(index.describe_index_stats())
    #get embeddings from db
    embeddings = transcript_collection.find_one({"_id": ObjectId(transcript_id)},{"_id":0, "embed_index":1})
    #insert embeddings in index
    embeddings_list = embeddings.get("embed_index")
    for i in range (0,len(embeddings_list[0])):
        embeddings_list[0][i] = tuple(embeddings_list[0][i])
    print(type(embeddings_list[0][0][2]))
    index.upsert(embeddings_list)
    #retrieve results of search
    #xq = model.encode(query).tolist()
    # results = index.query(xq, top_k=3, include_metadata=True)
    # print(results)
    # bestStart = results['matches'][0]['metadata']['startTime']
    # bestEnd = results['matches'][0]['metadata']['endTime']
    # bestAnswer = results['matches'][0]['metadata']['ref']
    # #start and end times are in seconds
    # print(bestStart)
    # print(bestEnd)
    # print(bestAnswer)
    return "done"
     
    
    

if __name__ == "__main__":
    app.run(debug=True)

