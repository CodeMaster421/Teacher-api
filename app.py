from flask import Flask, request, jsonify, send_from_directory
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv  # Add this import
from flask_cors import CORS
import json

load_dotenv()  # Add this line

app = Flask(__name__)
app.config.from_pyfile('config.py')
CORS(app)

# Set your OpenAI API key and assistant ID here
api_key = os.environ["OPENAI_API_KEY"]
assistant_id = os.environ["OPENAI_AI_ASSISTANT"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
API_KEY = "kristal-pdf"
pinecone_env = "us-east1"

PDF_PATHS = {
    "static/pdf/Singapore.pdf": "Singapore",
    "static/pdf/Abu_Dhabi.pdf": "Abu Dhabi",
    "static/pdf/Anti-Money Laundering and Sanctions Rules and Guidance.pdf": "Anti-Money Laundering and Sanctions Rules and Guidance",
    "static/pdf/Bank_Recovery_and_Resolution_Regulation_2018_20_December_2018.pdf": "Bank Recovery and Resolution Regulation 2018",
    "static/pdf/Captive Insurance Business Rules (CIB).pdf": "Captive Insurance Business Rules (CIB)",
    "static/pdf/COMMON REPORTING STANDARD REGULATIONS 2017.pdf": "Common Reporting Standard Regulations 2017",
    "static/pdf/Conduct of Business Rulebook (COBS).pdf": "Conduct of Business Rulebook (COBS)",
    "static/pdf/Fees Rules (FEES).pdf": "Fees Rules (FEES)",
    "static/pdf/FINANCIAL SERVICES AND MARKETS REGULATIONS 2015.pdf": "Financial Services And Markets Regulations 2015",
    "static/pdf/Fund Rules (FUNDS).pdf": "Fund Rules",
    "static/pdf/General Rulebook (GEN).pdf": "General Rulebook",
    "static/pdf/GLOSSARY (GLO).pdf": "Glossary",
    "static/pdf/Islamic Finance Rules (IFR).pdf": "Islamic Finance Rules",
    "static/pdf/Market Infrastructure Rulebook (MIR).pdf": "Market Infrastructure Rulebook (MIR)",
    "static/pdf/Market Rules (MKT).pdf": "Market Rules (MKT)",
    "static/pdf/Prudential – Insurance Business (PIN).pdf": "Prudential - Insurance Business (PIN)",
    "static/pdf/Prudential – Investment, Insurance Intermediation and Banking Rules.pdf": "Investment, Insurance Intermediation and Banking Rules",
}  # Add your PDF paths here

# Initialize Pinecone
index_name = "kristal-ai"
def convert_documents_to_dicts(data):
    new = []
    for doc in data:
        new.append({'page_content': doc.page_content, 'metadata': doc.metadata})
    return new

def load_or_create_vectorstore():
    embeddings=OpenAIEmbeddings(api_key=api_key)
    print(PineconeVectorStore(index_name=index_name, embedding=embeddings))
    return PineconeVectorStore(index_name=index_name, embedding=embeddings)

vectorstore = load_or_create_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k":2})
# Set OpenAI client, assistant, and thread
def load_openai_client_and_assistant():
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    my_assistant = client.beta.assistants.retrieve(assistant_id)
    return client, my_assistant

client, my_assistant = load_openai_client_and_assistant()

# Check in loop if assistant AI parses our request
def wait_on_run(run, thread):
    while run.status in ["queued", "in_progress"]:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        time.sleep(0.5)
    return run

# Initiate assistant AI response
def get_assistant_response(thread_id, user_input=""):
    message = client.beta.threads.messages.create(thread_id=thread_id, role="user", content=user_input)
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
    run = wait_on_run(run, client.beta.threads.retrieve(thread_id=thread_id))
    messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc", after=message.id)
    message_content = messages.data[0].content[0].text
    annotations = message_content.annotations
    citations = []
    # Iterate over the annotations and add footnotes
    for index, annotation in enumerate(annotations):
        # Replace the text with a footnote
        message_content.value = message_content.value.replace(annotation.text, '')
    # Add footnotes to the end of the message before displaying to user
    return message_content.value, retriever.invoke(message_content.value)

# Endpoint to create a new thread
@app.route('/create', methods=['POST', 'GET'])
def create_thread():
    try:
        thread = client.beta.threads.create()
        response = jsonify({"thread_id": thread.id})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Endpoint to handle user input and get response
@app.route('/ask', methods=['POST', 'GET'])
def ask():
    try:
        data = request.json
        thread_id = data.get('thread_id')
        user_input = data.get('query')
        if not thread_id or not user_input:
            return jsonify({"error": "Missing thread ID or query parameter"}), 400
        
        response, context = get_assistant_response(thread_id, user_input)
        if context is not None:
            context = convert_documents_to_dicts(context)
            response = jsonify({"response": response, "context": context})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
        else:
            return jsonify({"error": response["error"]}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

json_data={}

@app.route('/get_data', methods=['GET'])
def get_data():
    response = jsonify(json_data)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/update_data', methods=['POST'])
def update_data():
    # Check if API key is provided in the request headers
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401
    
    new_data = request.json
    global json_data
    json_data = new_data
    # Save the updated data to a file
    with open('data.json', 'w') as file:
        json.dump(new_data, file)

    response=jsonify('JSON file updated successfully')
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

# Endpoint to serve the HTML page
@app.route('/')
def index():
    return send_from_directory('', 'static/html/teacher.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
