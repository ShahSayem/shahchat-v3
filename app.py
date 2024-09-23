import os
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredExcelLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('Google_API_KEY'))

# Load Gemini Pro model
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])


class GeminiChatbot:
    def __init__(self):
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    # Get response from Gemini model
    def get_gemini_response(self, question):
        response = chat.send_message(question, stream=True)
        return response

    # Scrape a webpage and convert it into a LangChain Document object
    def scrape_website_as_document(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n")
            webpage_document = Document(page_content=text, metadata={"source": url})

            return webpage_document
        
        except Exception as e:
            return None

    # Load PDF files as LangChain Document objects
    def load_pdf_documents(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        return documents

    # Load Excel files as LangChain Document objects
    def load_excel_documents(self, excel_path):
        loader = UnstructuredExcelLoader(excel_path)
        documents = loader.load()
        
        return documents

    # Load TXT files as LangChain Document objects
    def load_txt_documents(self, txt_path):
        loader = TextLoader(txt_path)
        documents = loader.load()

        return documents

    # Split documents into chunks and embed them using HuggingFace
    def process_documents(self):
        final_documents = self.text_splitter.split_documents(self.documents)
        db = FAISS.from_documents(final_documents, self.embedding_function)

        return db


