import streamlit as st
import ollama
import PyPDF2
from docx import Document
import openpyxl
import os
import pandas as pd
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import io
from dotenv import load_dotenv
import logging
import time
import warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Suppress warnings to keep logs clean
warnings.filterwarnings("ignore")

# Load environment variables from .env file for configuration
load_dotenv()
TESSERACT_PATH = os.getenv("TESSERACT_PATH")  
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))  
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP'))
N_CHUNKS = int(os.getenv('N_CHUNKS'))

# Configure logging to file and console for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()  # Output logs to console
    ]
)
logger = logging.getLogger(__name__)

# Initialize the embedding model for vectorizing text
@st.cache_resource
def load_embeddings_model():
    """
    Loads the HuggingFace embedding model for text vectorization.
    """
    logger.info("Initializing embedding model...")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    logger.info(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")
    return embeddings

embeddings = load_embeddings_model()

def extract_text(file):
    """
    Extracts text from uploaded files (.txt, .pdf, .docx, .xlsx) using appropriate libraries.
    For PDFs, attempts PyPDF2 extraction first, falling back to OCR if needed.
    """
    start_time = time.time()
    file_extension = os.path.splitext(file.name)[1].lower()
    logger.info(f"Starting text extraction from file: {file.name} (type: {file_extension})")
    text = ""
    
    try:
        if file_extension == ".txt":
            text = file.read().decode("utf-8")
            logger.info(f"Text extracted from .txt file: {len(text)} characters")
        elif file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            logger.info(f"Extracted {len(text)} characters using PyPDF2 from .pdf")
            
            # Fallback to OCR if text is empty or too short
            if not text.strip() or len(text.strip()) < 100:
                logger.info("Text empty or too short, applying OCR")
                text = ""
                file.seek(0)
                images = convert_from_bytes(file.read())
                for i, img in enumerate(images):
                    page_text = pytesseract.image_to_string(img, lang="eng+rus")
                    text += page_text + "\n"
                    logger.info(f"OCR: Extracted {len(page_text)} characters from page {i+1}")
        elif file_extension == ".docx":
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
            logger.info(f"Text extracted from .docx file: {len(text)} characters")
        elif file_extension == ".xlsx":
            workbook = openpyxl.load_workbook(file)
            for sheet in workbook.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    text += " ".join([str(cell) for cell in row if cell is not None]) + "\n"
            logger.info(f"Text extracted from .xlsx file: {len(text)} characters")
        else:
            logger.error(f"Unsupported file format: {file_extension}")
            st.error("Only .txt, .pdf, .docx, and .xlsx files are supported")
            return None
        logger.info(f"Text extraction completed in {time.time() - start_time:.2f} seconds")
        return text.strip() if text.strip() else None
    except Exception as e:
        logger.error(f"Error processing file {file.name}: {str(e)}")
        st.error(f"Error processing file: {str(e)}")
        return None

def create_vector_store(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Creates a FAISS vector store from text by splitting it into chunks and embedding them.
    """
    start_time = time.time()
    logger.info("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Text split into {len(chunks)} chunks")
    
    documents = [Document(page_content=chunk) for chunk in chunks]
    logger.info("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(documents, embeddings)
    logger.info(f"Vector store created in {time.time() - start_time:.2f} seconds")
    return vector_store

def query_model(vector_store, question, n_chunks=N_CHUNKS):
    """
    Performs a Retrieval-Augmented Generation (RAG) query using the vector store and question.
    """
    start_time = time.time()
    logger.info(f"Executing RAG query: {question[:50]}...")
    
    # Define the prompt template for the model
    prompt_template = """You are a helpful assistant that answers questions based on the provided document.
    Use only relevant information from the context. Answer concisely and to the point.
    Context: {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Set up the RAG chain
    retriever = vector_store.as_retriever(search_kwargs={"k": n_chunks})
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | ollama_chat
        | StrOutputParser()
    )
    
    try:
        answer = chain.invoke(question)
        logger.info(f"RAG query completed in {time.time() - start_time:.2f} seconds")
        return answer
    except Exception as e:
        logger.error(f"Error executing RAG query: {str(e)}")
        return f"Error processing query: {str(e)}"

def ollama_chat(inputs):
    """
    Interacts with the Ollama model to generate a response based on context and question.
    """
    context = "\n".join([doc.page_content for doc in inputs["context"]])
    prompt = inputs["question"]
    full_prompt = f"Document context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
    logger.info(f"Sending request to gemma3n model with context ({len(context)} characters)")
    response = ollama.chat(
        model="gemma3n",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document. Answer concisely and to the point."},
            {"role": "user", "content": full_prompt}
        ]
    )
    return response['message']['content']

# Initialize session state to store document text, vector store, and history
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit interface setup
st.title("Document Analysis Assistant (Gemma3n with RAG)")
st.write("Upload a document (.txt, .pdf, .docx, or .xlsx) and ask questions about its content.")

# File uploader for document input
uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf", "docx", "xlsx"])

# Process uploaded file
if uploaded_file is not None:
    logger.info(f"File uploaded: {uploaded_file.name}")
    document_text = extract_text(uploaded_file)
    if document_text:
        st.session_state.document_text = document_text
        st.session_state.vector_store = create_vector_store(document_text)
        st.success(f"Document successfully loaded! Text length: {len(document_text)} characters")
        logger.info(f"Document loaded, text length: {len(document_text)} characters")
    else:
        st.session_state.document_text = None
        st.session_state.vector_store = None
        logger.warning("Failed to extract text from file")

# Input field for user question
question = st.text_input("Ask a question about the document:")

# Button to submit the question
if st.button("Submit question", key="submit_question"):
    if st.session_state.vector_store and question:
        with st.spinner("Processing query..."):
            logger.info("Submit question button clicked")
            answer = query_model(st.session_state.vector_store, question, N_CHUNKS)
            # Save question and answer to history
            st.session_state.history.append({"Question": question, "Answer": answer})
            st.write("**Answer:**")
            st.write(answer)
            logger.info("Answer displayed in interface")
    else:
        st.error("Please upload a document and enter a question.")
        logger.warning("Error: Missing document or question")

# Display history of questions and answers
if st.session_state.history:
    st.markdown("---")
    st.subheader("History of Questions and Answers")
    history_df = pd.DataFrame(st.session_state.history)
    st.table(history_df)
    logger.info(f"Displayed history: {len(st.session_state.history)} entries")

# Button to clear history
if st.button("Clear history", key="clear_history"):
    st.session_state.history = []
    st.success("History cleared!")
    logger.info("Question and answer history cleared")

# Instructions for using the app
st.markdown("---")
st.markdown("**Instructions:**\n1. Upload a document in .txt, .pdf, .docx, or .xlsx format.\n\
            2. Enter a question related to the document's content.\n\
            3. Click 'Submit question' to get an answer from the Gemma3n model.\n\
            4. View the history of questions and answers below.\n\
            5. Click 'Clear history' to remove all records.")

# Display recent logs in the interface
st.markdown("---")
st.subheader("Application Logs")
if os.path.exists("app.log"):
    with open("app.log", "r", encoding="utf-8") as log_file:
        logs = log_file.readlines()
        # Show the last 10 log entries
        st.text_area("Recent logs", value="".join(logs[-10:]), height=200, disabled=True)
    logger.info("Logs displayed in interface")
else:
    st.text("Logs not yet available")
    logger.info("Log file not yet created")