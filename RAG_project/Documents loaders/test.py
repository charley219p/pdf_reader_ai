from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

# Use a relative path since notes.txt is in the same directory as main.py
loader = TextLoader("notes.txt")

docs = loader.load()
print(f"Successfully loaded {len(docs)} document(s).")
print(docs[0])