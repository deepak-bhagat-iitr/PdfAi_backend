import fitz  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import os
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer


nlp = spacy.load('en_core_web_sm')


model = SentenceTransformer('all-mpnet-base-v2')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

SIMILARITY_THRESHOLD = 0.3  

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop])

def process_query(query):
    query = preprocess_text(query)
    query_vector = model.encode(query)
    return query_vector

def find_best_match(text, query_vector):
    sentences = [sent.text for sent in nlp(text).sents]
    preprocessed_sentences = [preprocess_text(sent) for sent in sentences]
    sentence_vectors = model.encode(preprocessed_sentences)
    similarities = cosine_similarity([query_vector], sentence_vectors)[0]
    best_match_index = np.argmax(similarities)
    return sentences[best_match_index], similarities[best_match_index]

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _set_response(self, code=200, message=None):
        self.send_response(code)
        self._set_headers()
        if message:
            self.wfile.write(json.dumps({"message": message}).encode())
        else:
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_headers()

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self._set_headers()
            with open('index.html', 'rb') as file:
                self.wfile.write(file.read())
        else:
            self._set_response(404, "Not Found")

    def do_POST(self):
        global extracted_text
        if self.path == '/upload':
            try:
                content_type = self.headers['Content-Type']
                if 'multipart/form-data' not in content_type:
                    self._set_response(400, "Content-Type must be multipart/form-data")
                    return

                content_length = int(self.headers['Content-Length'])
                body = self.rfile.read(content_length)
                boundary = content_type.split('=')[1].encode()
                parts = body.split(b'--' + boundary)

                for part in parts:
                    if part:
                        part = part.strip()
                        if not part:
                            continue

                        headers, content = part.split(b'\r\n\r\n', 1)
                        headers = headers.decode('utf-8')
                        disposition = headers.split('\r\n')[0]

                        if 'filename' in disposition:
                            filename = disposition.split('filename=')[1].strip('"')
                            file_path = os.path.join(UPLOAD_FOLDER, filename)

                            with open(file_path, 'wb') as output_file:
                                output_file.write(content.strip())

                            extracted_text = extract_text_from_pdf(file_path)

                            self._set_response(200, "File uploaded successfully")
                            return

                self._set_response(400, "File part not found in the request")
            except Exception as e:
                logging.error(f"Error handling upload: {e}")
                self._set_response(500, f"Server error: {str(e)}")
        elif self.path == '/':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                question = data.get('question', '')
                if extracted_text:
                    query_vector = process_query(question)
                    best_match, similarity = find_best_match(extracted_text, query_vector)

                    if similarity < SIMILARITY_THRESHOLD:
                        response = {"answer": "This Context is not related to pdf", "similarity": float(similarity)}
                    else:
                        response = {"answer": best_match, "similarity": float(similarity)}

                    self._set_response(200)
                    self.wfile.write(json.dumps(response).encode())
                else:
                    self._set_response(400, "No PDF uploaded or processed")
            except Exception as e:
                logging.error(f"Error handling question: {e}")
                self._set_response(500, f"Server error: {str(e)}")

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info(f'Starting httpd server on port {port}...')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd server...')

if __name__ == "__main__":
    extracted_text = None
    run()
