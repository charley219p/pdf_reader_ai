from langchain_community.document_loaders import PyPDFLoader
data = PyPDFLoader("journal paper1.pdf")
docs = data.load()
print(docs)