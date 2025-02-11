import streamlit as st
import logging
from langchain_ollama import OllamaLLM
import chromadb
import os
import requests
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
import fitz  
import json
import time
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO)

chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

class EmbeddingFunction:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input)

embedding = EmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
)


st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #f0f4f7;
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .header {
            font-size: 36px;
            color: #333;
            font-weight: bold;
            padding: 20px 0;
            text-align: center;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        .button:hover {
            background-color: #45a049;
        }
        .content {
            padding: 20px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .footer {
            font-size: 14px;
            color: #888;
            text-align: center;
            padding: 10px 0;
            margin-top: 30px;
            border-top: 1px solid #eee;
        }
        .query-section input {
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .query-section {
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)




def extract_pdf_text(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    pdf_document.close()
    if not text.strip():
        raise ValueError("The uploaded PDF is empty or contains no extractable text.")
    return text

def search_duckduckgo(query, num_results=5):
    results = []
    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=num_results)
            for result in search_results:
                title = result.get('title', 'No Title Available')
                url = result.get('href', 'No URL Available')
                description = result.get('body', 'No Description Available')

                results.append({
                    'title': title,
                    'url': url,
                    'description': description
                })

        if not results:
            raise ValueError("No search results found.")

    except Exception as e:
        logging.error(f"DuckDuckGo search failed: {e}")
        st.error("DuckDuckGo search failed. Please check your query or internet connection.")
        return []

    return results


def chat_with_ollama(query, model_name="llama3.2:1b"):
    """Query Ollama to get a response."""
    llm = OllamaLLM(model=model_name)
    response = llm.invoke(query).strip()
    return response


def save_to_chromadb(query, results):
    """Save results to ChromaDB."""
    doc_id = f"doc{len(collection.get().get('documents', [])) + 1}"
    collection.add(documents=[f"Query: {query}\nResults: {results}"], ids=[doc_id])
    logging.info(f"Document added to ChromaDB with ID {doc_id}")


def save_query_to_json(query, answer):
    """Save query and response to query_history.json."""
    timestamp = time.time()
    data = {
        "query": query,
        "answer": answer,
        "timestamp": timestamp
    }
    

    try:
        with open("query_history.json", "r") as file:
            history = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []


    history.append(data)


    with open("query_history.json", "w") as file:
        json.dump(history, file, indent=4)
    
    logging.info(f"Query and answer saved to query_history.json: {data}")


def search_and_respond(query):
    
    documents = collection.get()["documents"]
    relevant_documents = []
   
    for doc in documents:
        if query.lower() in doc.lower(): 
            relevant_documents.append(doc)
    
    if relevant_documents:
        
        context = " ".join(relevant_documents)
        query_with_context = f"{query} Context: {context}"

   
        response = chat_with_ollama(query_with_context, "llama3.2:1b")

       
        st.subheader("Response from Ollama (Document-based):")
        st.write(response)

       
        save_query_to_json(query, response)

        return response
    else:
        st.write("No relevant documents found in ChromaDB, searching the web...")
    
    
    search_results = search_duckduckgo(query)

    if search_results:
       
        st.subheader("Search Results from DuckDuckGo")

        
        sources = "\n".join([f"{res['title']} ({res['url']}): {res['description']}" for res in search_results])

        
        st.write(sources)

       
        response_from_chroma = chat_with_ollama(f"{query} {sources}", "llama3.2:1b")
        st.write(response_from_chroma)

       
        save_to_chromadb(query, search_results)

        
        save_query_to_json(query, response_from_chroma)

        return response_from_chroma
    else:
        st.write("No relevant information found in DuckDuckGo.")
        return None


def display_documents():
    """Display all documents from the ChromaDB collection."""
    documents = collection.get()["documents"]
    
    if documents:
        st.subheader("Saved Documents:")
        for idx, doc in enumerate(documents, start=1):
            st.text_area(f"Document {idx}", doc, height=200)
    else:
        st.write("No documents found in the database.")


def visualize_wordcloud():
    """Create and display a word cloud for each document in ChromaDB."""
    documents = collection.get()["documents"]
    
    if not documents:
        st.warning("No documents available to generate word clouds.")
        return
    
    for idx, doc in enumerate(documents, start=1):
        st.subheader(f"Word Cloud for Document {idx}")
        

        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(doc)
        
        
        st.image(wordcloud.to_array(), use_container_width=True)


menu = st.sidebar.selectbox(
    "Choose an action", 
    ["Home Page", "Show Word Cloud", "Upload and Add Document", "Ask Ollama a Question", "View Saved Documents"]
)


st.markdown('<div class="header">RAG System with Ollama</div>', unsafe_allow_html=True)


if menu == "Home Page":
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.subheader("Welcome to the RAG System with Ollama")
    st.markdown("""
        **Introduction**:
        
        This application uses Retrieval-Augmented Generation (RAG) and Ollama to provide AI-powered responses. 
        Users can ask questions, query stored documents, upload files, and even generate word clouds based on the 
        data in ChromaDB.

        **What You Can Do**:
        - **Ask a Question**: Get responses based on documents stored in the database or from external sources.
        - **Upload and Add Document**: Upload your own documents (PDF/TXT files) for Ollama to query.
        - **View Saved Documents**: View documents that have been uploaded to the ChromaDB.
        - **Show Word Cloud**: Generate word clouds based on your stored documents to analyze word frequencies.
        
        **How to Use**:
        1. Select "Ask a Question" to type a query and receive a response.
        2. Select "Upload and Add Document" to add documents to the system.
        3. View your saved documents under "View Saved Documents".
        4. Create a word cloud of your documents' contents in the "Show Word Cloud" section.

        **Key Features**:
        - **Real-time Information**: Get real-time answers based on your stored data or the web.
        - **Document Uploading**: Upload your documents and query them for specific answers.
        - **Interactive Interface**: A clean and simple interface to interact with the app seamlessly.
    """)
    st.markdown('</div>', unsafe_allow_html=True)


elif menu == "Show Word Cloud":
    visualize_wordcloud()

elif menu == "Upload and Add Document":
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:
        text = extract_pdf_text(uploaded_file) if uploaded_file.type == "application/pdf" else uploaded_file.read().decode("utf-8").strip()
        if text:
            doc_id = f"doc{len(collection.get().get('documents', [])) + 1}"
            collection.add(documents=[text], ids=[doc_id])
            st.success(f"Document added to ChromaDB with ID {doc_id}")

elif menu == "Ask Ollama a Question":
    st.markdown('<div class="query-section">', unsafe_allow_html=True)
    query = st.text_input("Ask a question")
    if query:
        response = search_and_respond(query)  
        if response:
            st.write("Response:", response)
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "View Saved Documents":
    display_documents()


st.markdown('<div class="footer">Built with 💖 by Qyzdar </div>', unsafe_allow_html=True)