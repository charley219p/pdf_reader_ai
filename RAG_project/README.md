# 📚 PDF Reader AI – RAG Document Question Answering

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to ask natural language questions about their documents. The project uses **Google Gemini Embeddings** for semantic indexing, **ChromaDB** as the vector database, and **Mistral AI** to generate accurate, context-aware answers based only on the retrieved document content.

---

## 🚀 Features

* 📄 Document-based Question Answering
* 🔍 Semantic Search using Vector Embeddings
* 🗂️ ChromaDB Vector Database
* 🧠 Google Gemini Embeddings
* 🤖 Mistral AI for Response Generation
* ⚡ Max Marginal Relevance (MMR) Retrieval
* 🔒 Context-aware responses to reduce AI hallucinations
* 🛠️ Built with LangChain

---

## 🏗️ Tech Stack

* Python
* LangChain
* Google Gemini Embeddings
* Mistral AI
* ChromaDB
* python-dotenv

---

## 📂 Project Structure

```text
RAG_project/
│── chroma_db/           # Vector database
│── main.py              # Main RAG application
│── requirements.txt     # Project dependencies
│── .env                 # API keys (not included)
│── .gitignore
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/charley219p/pdf_reader_ai.git
cd pdf_reader_ai
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root and add your API keys:

```env
GOOGLE_API_KEY=your_google_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

> **Important:** Never commit your `.env` file to GitHub.

---

## ▶️ Run the Project

```bash
python main.py
```

Once the application starts, enter your questions in the terminal.

Example:

```text
You: What is Retrieval-Augmented Generation?
AI: Retrieval-Augmented Generation (RAG) is...
```

Type `0` to exit the application.

---

## 🧠 How It Works

1. Documents are converted into vector embeddings using Google Gemini Embeddings.
2. The embeddings are stored in ChromaDB.
3. The retriever performs semantic search using Max Marginal Relevance (MMR).
4. The most relevant document chunks are passed to Mistral AI.
5. Mistral generates an answer using only the retrieved context.

---

## 📌 Future Improvements

* PDF upload interface
* Multi-document support
* Chat history
* Streamlit web application
* Source citations
* Hybrid search (Keyword + Vector)
* Support for multiple embedding models

---

## 🤝 Contributing

Contributions, feature requests, and suggestions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub. It helps others discover the project and motivates further development.
