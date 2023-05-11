from flask import Flask, jsonify, request, render_template
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import openai
from utils import distances_from_embeddings, indices_of_nearest_neighbors_from_distances



app = Flask(__name__, template_folder='static')
load_dotenv()
#connect to db and get collections
db_uri = os.getenv("MONGO_HOST")
db_name = os.getenv("DB_NAME")
db_port = os.getenv("MONGO_PORT")
client = MongoClient(host=db_uri, port=int(db_port), server_api=ServerApi('1'))
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key       
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
        print(e)

db = client[db_name]
synthesis_collection = db["synthesis"]



@app.route(f"/search", methods=["GET"])
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
        results.append({'_id':str(search["_id"]),'title':search["title"],'text':search["text"],'tags':search["tags"] })
    print(results)
    print(search_result)
    # Return synthesized document
    return results


    
    

if __name__ == "__main__":
    app.run(debug=True)

