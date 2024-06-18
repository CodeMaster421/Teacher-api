from flask import Flask, request, jsonify, send_from_directory
import time
from openai import OpenAI
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = Flask(__name__)
app.config.from_pyfile('config.py')

# Set your OpenAI API key and assistant ID here
api_key = os.environ["OPENAI_API_KEY"]
assistant_id = os.environ["OPENAI_AI_ASSISTANT"]

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
VECTORSTORE_PATH = "static/vectorstore"

def convert_documents_to_dicts(data):
    new = []
    for doc in data:
        new.append({'page_content': doc.page_content, 'metadata': doc.metadata})
    return new

def load_or_create_vectorstore(pdf_paths, path):
    if os.path.exists(path):
        print("Database loaded from memory")
        return Chroma(persist_directory=path, embedding_function=OpenAIEmbeddings())
    else:
        print("Creating database...")
        all_splits = []
        for pdf_path, _ in pdf_paths.items():
            loader = PyPDFLoader(pdf_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(loader.load())
            for split in splits:
                split.metadata['pdf_name'] = os.path.basename(pdf_path)  # Add PDF name to metadata
            all_splits.extend(splits)
        return Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory=path)

vectorstore = load_or_create_vectorstore(PDF_PATHS, VECTORSTORE_PATH)
retriever = vectorstore.as_retriever()

# Set OpenAI client, assistant, and thread
def load_openai_client_and_assistant():
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
    try:
        message = client.beta.threads.messages.create(thread_id=thread_id, role="user", content=user_input)
        run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
        run = wait_on_run(run, client.beta.threads.retrieve(thread_id=thread_id))
        messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc", after=message.id)
        return messages.data[0].content[0].text.value, retriever.invoke(user_input)
    except:
        return {"error": f"No thread found with id '{thread_id}'."}, None

# Endpoint to create a new thread
@app.route('/create', methods=['POST'])
def create_thread():
    thread = client.beta.threads.create()
    return jsonify({"thread_id": thread.id})

# Endpoint to handle user input and get response
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    thread_id = data.get('thread_id')
    user_input = data.get('query')
    response, context = get_assistant_response(thread_id, user_input)
    if context is not None:
        context = convert_documents_to_dicts(context)
        return jsonify({"response": response, "context": context})
    else:
        return jsonify({"error": response["error"]}), 404

# Endpoint to serve the HTML page
@app.route('/')
def index():
    return send_from_directory('', 'static/html/teacher.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
