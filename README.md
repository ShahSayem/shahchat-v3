# Gemini LLM Chatbot with File & URL Processing

## Overview
This project implements a chatbot powered by Google's Gemini-1.5-flash and Paraphrase-MiniLM-L3-v2 large language models (LLM). The chatbot can process user queries by leveraging uploaded documents (PDF, Excel, TXT) and web pages, providing intelligent responses using context retrieved from these sources.

## Features
- Supports multiple document formats (PDF, Excel, TXT) as knowledge sources.
- Scrapes web pages and integrates their content into the chatbot's responses.
- Uses LangChain for document processing and embedding.
- FAISS-based vector search for efficient document retrieval.
- Streamlit-based UI for easy interaction.

## File Structure
```
|-- app.py          # Core logic for Gemini-based chatbot & document processing
|-- client.py       # Streamlit UI for chatbot interaction
|-- requirements.txt # Dependencies required for running the chatbot
```

## Installation

### 1. Clone the Repository
```sh
$ git clone https://github.com/ShahSayem/shahchat-v3.git
```

### 2. Create a Virtual Environment (Optional but Recommended)
```sh
$ python -m venv venv
$ source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```sh
$ pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file and add your Google Gemini API key:
```sh
Google_API_KEY=<your-api-key>
```

## Usage

### 1. Run the Chatbot
```sh
$ streamlit run client.py
```

### 2. Upload Files & Add URLs
- Use the sidebar to upload PDF, Excel, or TXT files.
- Enter website URLs to scrape relevant content.
- Click the `Process files & URLs` button to preprocess documents.

### 3. Ask Questions
- Type your query in the input box and get AI-generated responses based on the provided data.

## Dependencies
The chatbot requires the following Python packages:
- `langchain`
- `langchain-community`
- `python-dotenv`
- `streamlit`
- `PyPDF2`
- `openpyxl`
- `bs4`
- `faiss-cpu`
- `requests`
- `unstructured`
- `beautifulsoup4`
- `sentence-transformers`
- `google-generativeai`

## License
This project is licensed under the MIT License.
