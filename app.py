import hashlib
import os
from flask import Flask, Response, render_template, request, jsonify, stream_with_context
import pickle
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from faster_whisper import WhisperModel
import tempfile

app = Flask(__name__, static_folder='temp', static_url_path='/temp')

# Global variables
pdf_documents = []   # all documents from all PDFs
vector_store = None
llm = ChatOllama(model="gemma3:4b")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")


CACHE_DIR = "cache"

# Helper to process a single PDF
def process_pdf(file):
    pdf_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(pdf_path)

    doc = fitz.open(pdf_path)
    page_data = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        images = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = base_image["ext"]
            img_filename = f"{file.filename}_page{page_num+1}_img{img_index}.{ext}"
            img_path = os.path.join("temp", img_filename)
            with open(img_path, "wb") as f_img:
                f_img.write(img_bytes)
            images.append(img_path)

        page_data.append({
            "text": text,
            "images": images,
            "page_num": page_num,
            "pdf_filename": file.filename
        })
    doc.close()
    return page_data

# Helper to create vector store for a set of documents
def create_vector_store(documents):
    embeddings = embedding_model.encode([doc['text'] for doc in documents])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Retriever that searches across all loaded PDFs
def get_retriever():
    def retriever_fn(query, k=2):
        query_emb = embedding_model.encode([query])
        distances, indices = vector_store.search(query_emb, k)
        return [pdf_documents[i] for i in indices[0]]
    return retriever_fn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global pdf_documents, vector_store

    if 'pdfs' not in request.files:
        return jsonify({'error': 'No PDF files provided'}), 400

    files = request.files.getlist('pdfs')
    os.makedirs(CACHE_DIR, exist_ok=True)

    new_documents = []

    for file in files:
        file_bytes = file.read()
        file.seek(0)
        pdf_hash = hashlib.sha256(file_bytes).hexdigest()
        cache_path = os.path.join(CACHE_DIR, pdf_hash)
        os.makedirs(cache_path, exist_ok=True)

        # Load from cache if exists
        if os.path.exists(os.path.join(cache_path, "documents.pkl")) and os.path.exists(os.path.join(cache_path, "vector_store.faiss")):
            with open(os.path.join(cache_path, "documents.pkl"), "rb") as f:
                docs = pickle.load(f)
            index = faiss.read_index(os.path.join(cache_path, "vector_store.faiss"))
        else:
            # Process PDF
            page_data = process_pdf(file)
            docs = page_data
            index = create_vector_store(docs)
            with open(os.path.join(cache_path, "documents.pkl"), "wb") as f:
                pickle.dump(docs, f)
            faiss.write_index(index, os.path.join(cache_path, "vector_store.faiss"))

        pdf_documents.extend(docs)

    # Rebuild combined vector store for all PDFs
    vector_store = create_vector_store(pdf_documents)

    return jsonify({'message': 'PDFs processed successfully', 'pdf_filenames': [f.filename for f in files]})

from flask import json

@app.route('/ask', methods=['POST'])
def ask_question():
    global vector_store, pdf_documents
    if vector_store is None:
        return jsonify({'error': 'No PDFs uploaded yet'}), 400

    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    retriever = get_retriever()
    retrieved_docs = retriever(question)

    context_text = "\n\n".join([doc['text'] for doc in retrieved_docs])

    template = """Answer the question based only on the following context and answer should be in html format enclosed within <div> tag. Do not use markdown formatting or styles:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm | StrOutputParser()

    def generate():
        # Stream answer chunks
        for chunk in rag_chain.stream({"context": context_text, "question": question}):
            yield chunk
        # After answer is done, send sources separately
        sources = [
            {
                "pdf_filename": doc['pdf_filename'],
                "page_num": doc['page_num'] + 1  # 1-indexed page
            }
            for doc in retrieved_docs
        ]
        yield json.dumps({"type": "sources", "content": sources})

    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        audio_file.save(temp_audio.name)
        temp_audio_path = temp_audio.name

    try:
        segments, _ = whisper_model.transcribe(temp_audio_path, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
    finally:
        os.remove(temp_audio_path)

    return jsonify({'transcription': transcription})

if __name__ == '__main__':
    app.run(debug=True)
