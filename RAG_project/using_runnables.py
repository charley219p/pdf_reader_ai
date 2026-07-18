from dotenv import load_dotenv
from operator import itemgetter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# -------------------------------
# Load and Split PDF
# -------------------------------
loader = PyPDFLoader("Documents loaders/journal paper1.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300
)

chunks = splitter.split_documents(documents)

print(f"Pages Loaded: {len(documents)}")
print(f"Chunks Created: {len(chunks)}")

# -------------------------------
# Embedding Model
# -------------------------------
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

# -------------------------------
# Create / Load Vector Store
# -------------------------------
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print(f"Stored Documents: {vectorstore._collection.count()}")

# -------------------------------
# Retriever
# -------------------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# -------------------------------
# LLM
# -------------------------------
llm = ChatMistralAI(
    model_name="mistral-small-latest"
)

# -------------------------------
# Prompt
# -------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context to answer the user's question.

If the answer is not present in the context,
reply exactly:

"I could not find the answer in the document."

Context:
{context}
"""
        ),
        (
            "human",
            "{question}"
        )
    ]
)

# -------------------------------
# Function to Convert Documents
# -------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------
# Runnable Chain
# -------------------------------
chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question")
    }
    | prompt | llm | StrOutputParser()
)

# -------------------------------
# Chat Loop
# -------------------------------
print("\nRAG Chatbot Ready!")
print("Type 0 to exit.\n")

while True:

    query = input("You: ")

    if query == "0":
        break

    answer = chain.invoke(
        {
            "question": query
        }
    )

    print("\nAI:", answer)