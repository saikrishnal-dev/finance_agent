import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Verify data file exists
pdf_path = os.path.join("data", "report.pdf")
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Could not find {pdf_path}. Please place your PDF in the 'data' folder and name it 'report.pdf'.")

print(f"Loading {pdf_path}...")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} pages successfully.")

# 2. Split with table-safe parameters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks.")

# 3. Create Vector Store with Ollama Embeddings
print("Embedding and building Chroma vector database with nomic-embed-text (this may take a minute)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Vector store saved to ./chroma_db!")

# 4. Test Retriever with a sample query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
test_query = "What are the primary risk factors or revenue figures?"
print(f"\nTesting retriever with query: '{test_query}'...")
results = retriever.invoke(test_query)

print("\n=== RETRIEVER RESULTS ===")
for idx, doc in enumerate(results):
    page_num = doc.metadata.get("page", "Unknown")
    print(f"\n--- Result {idx + 1} (Page {page_num}) ---")
    print(doc.page_content[:250].strip() + "...\n")