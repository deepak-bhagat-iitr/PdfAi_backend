# PDF Answering AI

## Project Overview
PDF Answering AI is an intelligent system designed to extract text from PDF files and match user queries with the most relevant sections of the extracted text. The system utilizes natural language processing (NLP) techniques and machine learning models to achieve accurate query responses.

## Installation Instructions
To set up the environment and run the project, follow these steps:

### Step-by-Step Installation
1. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv myenv
   ```

2. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source myenv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install PyMuPDF numpy scikit-learn sentence-transformers spacy requests
   ```

4. **Download spaCy Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage
### Uploading PDFs
- Click on the upload button to select and upload a PDF file.

### Asking Queries
- After uploading a PDF, enter your question in the provided text box.
- Click "Submit" to see the AI's response.

## Dependencies
- `PyMuPDF`: For extracting text from PDF files.
- `numpy`: For numerical computations (used in some parts of the code, optional for your specific use case).
- `scikit-learn`: For cosine similarity calculations.
- `sentence-transformers`: For encoding sentences into vectors.
- `spacy`: For text preprocessing and tokenization.
- `requests`: For making HTTP requests (ensure you have a specific use case for this).

Ensure you have these dependencies installed and configured as per the installation instructions above.
