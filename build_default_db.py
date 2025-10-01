import os
import glob
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
CORPUS_PATH = "NHAI_corpus"  # Directory containing PDF files
INDEX_PATH = "default_faiss_index"

def build_vector_store():
    """Builds and saves a FAISS vector store from PDF documents in a directory."""
    pdf_files = glob.glob(os.path.join(CORPUS_PATH, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {CORPUS_PATH}. Please add some PDF documents and try again.")
        return

    all_docs = []
    for pdf_file in pdf_files:
        loader = UnstructuredPDFLoader(pdf_file)
        docs = loader.load()
        all_docs.extend(docs)

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    text_splitter = SemanticChunker(embeddings)
    split_docs = text_splitter.split_documents(all_docs)
    db = FAISS.from_documents(split_docs, embeddings)
    
    os.makedirs(INDEX_PATH, exist_ok=True)
    db.save_local(INDEX_PATH)
    print(f"Vector store built and saved to {INDEX_PATH} with {len(split_docs)} document chunks.")

if __name__=="__main__":
    build_vector_store()