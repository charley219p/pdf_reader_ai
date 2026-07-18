from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
load_dotenv()
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
loader = PyPDFLoader("Documents loaders/journal paper1.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 300
)
chunks = splitter.split_documents(documents)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db1"
)
print("Pages:", len(documents))
print("Chunks:", len(chunks))

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k": 4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)
print(vectorstore._collection.count())
print("Stored documents:", vectorstore._collection.count())
llm = ChatMistralAI(model_name = "mistral-small-latest")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
            use only the provided context to answer the question.
            if the answer is not present in the context,
            say: "I could not find the answer in the document.
            """
        ),
        (
            "human",
            """context
            {context}
            Question:
            {question}
            """
        )
    ]
)
print("rag system created")
print("press 0 to exit")
while True:
    query = input("You: ")
    if query == "0":
        break
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.invoke({
        "context":context,
        "question":query
    })
    response = llm.invoke(final_prompt)
    print(f"\n AI: {response.content}")