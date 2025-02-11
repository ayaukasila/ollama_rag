# AI-Powered Knowledge Base (RAG with Ollama & Streamlit)

## Overview
This project is a Collaborative AI-Powered Knowledge Base built using Streamlit, LangChain, and Ollama. It enables users to upload documents, ask questions, retrieve relevant information, and visualize insights using Retrieval-Augmented Generation (RAG) and ChromaDB.

## Features
- **Document Contribution System**
  - Upload PDFs or text files.
  - Extract content and store it in ChromaDB.
  
- **Natural Language Querying**
  - Ask questions about stored documents.
  - AI model (Ollama) generates responses based on indexed knowledge and web search.
  
- **Web Search Integration**
  - If no relevant documents are found in ChromaDB, the system searches for relevant information on DuckDuckGo.
  
- **Memory & Persistence**
  - Stores responses in JSON & ChromaDB for future reference.
  
- **Insight Visualization**
  - Generates word clouds for key topics within documents.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ayaukasila/ollama_rag.git
cd ollama_rag
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run 4.py
```

---

## Project Structure
```
📦 ap_final
├── README.md
├── requirements.txt
├── 4.py  # Main Streamlit App
├── chroma_db/  # ChromaDB Persistent Storage
└── query_history.json  # Stores past queries & answers
```

---

## How to Use
### Upload Documents
1. Navigate to "Upload and Add Document".
2. Upload a PDF or TXT file.
3. The file is processed & stored in ChromaDB.

### Ask a Question
1. Go to "Ask Ollama a Question".
2. Type your query (e.g., "Summarize the document").
3. The system searches for relevant documents in ChromaDB.
4. If no relevant documents are found, it fetches information from DuckDuckGo.
5. AI generates a response based on stored documents and web search results.

### View Saved Data
- View Saved Documents → See all uploaded files.
- Show Word Cloud → Visualize document contents.

---

## Requirements (`requirements.txt`)

```txt
streamlit
langchain_ollama
chromadb
duckduckgo_search
sentence-transformers
pymupdf  # For PDF extraction
requests
wordcloud
json
```

---

## Why This Project?
The AI-Powered Knowledge Base is designed to enhance document searchability and response accuracy by integrating **RAG-based AI models**. The ability to **combine stored data with real-time search** ensures that responses remain up-to-date and contextually accurate.

### Key Benefits:
- **Improved Knowledge Retrieval** - Searches indexed and live data.
- **AI-Powered Insights** - Provides intelligent responses.
- **Fallback Mechanism** - Searches DuckDuckGo when documents are insufficient.
- **Interactive Data Visualization** - Presents word clouds for content analysis.
- **Customizable & Expandable** - Built with modularity in mind.

---

## Future Improvements
- **Real-Time Collaboration** (Multi-user Access)
- **User Authentication & Role Management**
- **Advanced Data Visualization** (Graphs, Trends)
- **Deployment on Cloud Platforms (AWS, GCP)
- **Enhanced Semantic Search** for better document matching.

---

## Contributing
Contributions are welcome. If you would like to improve the project:
1. Fork the repository
2. Create a new branch (`feature-new`)
3. Commit your changes (`git commit -m "Added new feature"`)
4. Push to your fork (`git push origin feature-new`)
5. Open a Pull Request

---

## License
MIT License - Use, modify, and distribute freely.

