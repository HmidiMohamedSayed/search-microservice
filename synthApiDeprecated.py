from flask import Flask, jsonify, request, render_template
from synthesizer import Synthesizer
from db import Database
from dotenv import load_dotenv
from rabbitmq.receiver import Receiver
import os




app = Flask(__name__, template_folder='static')
load_dotenv()
db_uri = os.getenv("DB_URI")

db = Database(db_uri, 'myDataBase').db

collection = db["transcripts"]

synthesizer = Synthesizer()
receiver= Receiver()
receiver.receive()


@app.route('/', methods=['GET', 'POST'])
def index():
    transcripts = collection.find()
    transcript = transcripts[0] # assuming only one transcript in the database
    
    #convert ObjectId to string
    transcript['_id'] = str(transcript['_id'])
    transcript_dict = dict(transcript)

    if request.method == 'POST':
        return jsonify({"document": transcript_dict})
    
    return render_template('index.html')

@app.route("/synthesize", methods=["POST"])
def synthesize():
    
    # Retrieve transcript id and notes from request and convert to int
    transcript_id = int(request.get_json()['transcript_Id'])
    
    print(transcript_id)
    notes = request.get_json()['notes']
    print(notes)
    # Retrieve transcript from db with transcript_id
    transcript = collection.find({"id":transcript_id }).limit(1).next()  
    print(transcript)
    if not transcript:
        return jsonify({"message": "Transcript not found."}), 404
    
    print(transcript)
    #convert ObjectId to string
    transcript['_id'] = str(transcript['_id'])
    transcript_dict = dict(transcript)
   
    
    print(notes)

    if not notes:
        return jsonify({"message": "Notes not provided."}), 400

    # Synthesize document using Synthesizer
    synthesizer = Synthesizer()
    document = synthesizer.generate_summary(transcript_dict['text'], notes)

    # Return synthesized document
    return jsonify({"document": document})


    
    

if __name__ == "__main__":
    app.run(debug=True)

