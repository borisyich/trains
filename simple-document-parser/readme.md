# Simple-document-parser

The **Simple Documents Parser** is a Streamlit-based web application designed to extract text from documents in various formats (`.txt`, `.pdf`, `.docx`, `.xlsx`) and answer user questions about their content using a Retrieval-Augmented Generation (RAG) pipeline. It leverages the Gemma3n language model via Ollama, HuggingFace embeddings for text vectorization, and Chroma for vector storage. The application supports text extraction with OCR fallback for PDFs, maintains a question-answer history, and provides logging for debugging.

## Features
- **Supported Formats**: Upload and process `.txt`, `.pdf`, `.docx`, and `.xlsx` files.
- **Text Extraction**: Extracts text using libraries like PyPDF2, python-docx, and openpyxl, with OCR (Tesseract) fallback for scanned PDFs.
- **Question Answering**: Uses a RAG pipeline with Gemma3n to answer questions based on document content.
- **Session Management**: Stores document text, question history, and embeddings in Streamlit's session state.
- **Logging**: Comprehensive logging to a file (`app.log`) and console for debugging.
- **History**: Displays a table of previous questions and answers, with an option to clear the history.
- **User Interface**: Intuitive Streamlit interface with file uploader, question input, and log display.

## Requirements
- Python 3.8+
- Libraries:
  - `streamlit`
  - `ollama`
  - `PyPDF2`
  - `python-docx`
  - `openpyxl`
  - `pandas`
  - `pdf2image`
  - `pytesseract`
  - `Pillow`
  - `langchain`
  - `langchain-community`
  - `sentence-transformers`
  - `chromadb`
  - `nest_asyncio`
  - `python-dotenv`

Additionally, you need:
- **Tesseract OCR**: Installed on your system with the path specified in the `.env` file.
- **Ollama**: Installed and running with the `gemma3n` model available.
- **.env file**: Containing the following variables:
  ```plaintext
  TESSERACT_PATH=/path/to/tesseract
  CHUNK_SIZE=1000
  CHUNK_OVERLAP=200
  N_CHUNKS=8
  ```

## Installation
1. Clone the repository 
  ```bash
   git clone https://github.com/borisyich/trains.git
   cd trains/simple-document-parser
   ```
2. Create a `.env` file in the project root with:
   ```env
  TESSERACT_PATH=/path/to/tesseract
  CHUNK_SIZE=1000 # for example
  CHUNK_OVERLAP=200 # for example
  N_CHUNKS=8 # for example
   ```
3. Ensure Ollama is running with the Gemma3n model:
   ```bash
   ollama run gemma3n
   ```
## Usage
1. Open the application in your browser (typically at `http://localhost:8501`).
2. Upload a document (`.txt`, `.pdf`, `.docx`, or `.xlsx`).
3. Enter a question related to the document's content in the text input field.
4. Click **"Отправить вопрос"** to get an answer from the Gemma3n model.
5. View the question-answer history in the table below.
6. Use the **"Очистить историю"** button to clear the history.
7. Check the **"Логи приложения"** section for recent logs.

## File Structure
- `app.py`: Main application script containing the Streamlit app and logic.
- `app.log`: Log file for debugging and monitoring.
- `.env`: Configuration file for environment variables (e.g., Tesseract path, chunk settings).

## Notes
- The application uses a multilingual sentence transformer model (`paraphrase-multilingual-MiniLM-L12-v2`) for embeddings, supporting both English and Russian text.
- For PDFs, if PyPDF2 fails to extract sufficient text, the app falls back to OCR using Tesseract with English and Russian language support.
- The RAG pipeline splits documents into chunks (configurable via `CHUNK_SIZE` and `CHUNK_OVERLAP`) and retrieves the top `N_CHUNKS` relevant chunks for answering questions.
- Ensure the Ollama server is running and the `gemma3n` model is accessible before starting the app.
- Logs are stored in `app.log` and displayed in the UI (last 10 lines).

## Limitations
- Requires a local Ollama server and Tesseract installation.
- OCR may be slow for large PDFs with many pages.
- The `gemma3n` model must be available via Ollama.
- Only `.txt`, `.pdf`, `.docx`, and `.xlsx` formats are supported.

## Troubleshooting
- **Tesseract not found**: Ensure `TESSERACT_PATH` in `.env` points to the Tesseract executable.
- **Ollama errors**: Verify that the Ollama server is running and the `gemma3n` model is loaded.
- **File processing errors**: Check the `app.log` file for detailed error messages.
- **Slow performance**: Adjust `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `N_CHUNKS` in `.env` to optimize processing.
