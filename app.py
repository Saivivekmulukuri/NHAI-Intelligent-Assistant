import streamlit as st
import os
import shutil
import re
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter # Replaced with SemanticChunker
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- App Configuration ---
st.set_page_config(page_title="NHAI Intelligent Assistant", page_icon="🤖", layout="wide")

# --- Constants and API Key ---
# It is STRONGLY recommended to use st.secrets for your API key
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it to proceed.")
    st.info("Refer to the instructions on how to set up your secrets.toml file for local development or add the secret in the Streamlit Community Cloud settings.")
    st.stop()

LLM_MODEL_NAME = "qwen/qwen3-32b" 
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_INDEX_PATH = "default_faiss_index"

# --- State Management ---
if "user_vector_store" not in st.session_state:
    st.session_state.user_vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the NHAI Intelligent Assistant. You can ask me general questions about NHAI or upload your own documents for specific queries."}]

# --- Core Functions ---

@st.cache_resource
def load_default_vector_store():
    """Loads the pre-built default FAISS vector store."""
    if not os.path.exists(DEFAULT_INDEX_PATH):
        st.error("Default knowledge base not found. Please run 'build_default_db.py' first.")
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(DEFAULT_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def create_user_vector_store(_uploaded_files):
    """Creates a FAISS vector store from user-uploaded PDF files and cleans up temp files."""
    if not _uploaded_files:
        return None
    
    temp_dir = "temp_user_files"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with st.spinner("Processing your documents..."):
            all_docs = []
            for uploaded_file in _uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = UnstructuredPDFLoader(file_path)
                all_docs.extend(loader.load())

            # Using SemanticChunker for more intelligent splitting
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            text_splitter = SemanticChunker(embeddings)
            split_docs = text_splitter.split_documents(all_docs)
            db = FAISS.from_documents(split_docs, embeddings)
        st.success(f"Successfully processed {len(_uploaded_files)} document(s)!")
        return db
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def get_response_and_context(retriever, user_question):
    """
    Retrieves context, generates a thinking process, and a final answer from an LLM.
    """
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=LLM_MODEL_NAME, temperature=0.2)
    
    # 1. Retrieve relevant documents
    docs = retriever.get_relevant_documents(user_question)
    if not docs:
        return "I could not find any relevant information in the documents.", "", "No context retrieved."

    context_str = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # 2. Prompt for the "Thinking Process"
    thinking_prompt = ChatPromptTemplate.from_template("""
    Based on the following user question and context, explain your step-by-step reasoning for how you would form an answer. Focus on analyzing the context and planning the response. DO NOT provide the final answer itself, only the thought process.

    Context:
    {context}

    User Question: {input}
    """)
    thinking_chain = thinking_prompt | llm
    thinking_response = thinking_chain.invoke({"context": context_str, "input": user_question}).content

    # 3. Prompt for the "Final Answer"
    answer_prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant for the National Highways Authority of India (NHAI). Answer the user's question based ONLY on the context provided.
    Be concise and helpful. If the information is not in the context, state: 'I could not find an answer in the provided documents.'

    Context:
    {context}

    Question: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    final_answer_raw = document_chain.invoke({"context": docs, "input": user_question})

    # Clean the raw answer to remove the entire <think> block before returning
    final_answer = re.sub(r"<think>.*?</think>", "", final_answer_raw, flags=re.DOTALL).strip()

    return thinking_response, final_answer, context_str

def reset_conversation():
    """Resets the conversation and uploaded files."""
    st.session_state.user_vector_store = None
    st.session_state.messages = [{"role": "assistant", "content": "Session reset. How can I help you now?"}]
    # st.rerun() # This can be uncommented if you want an immediate clear and rerun

# --- UI Layout ---

st.title("NHAI Intelligent Assistant 🤖🛣️")

# Sidebar for document uploads and session control
with st.sidebar:
    st.header("Document-Specific Q&A")
    st.markdown("Have a question about a specific document? Upload it here.")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Process Documents"):
            st.session_state.user_vector_store = create_user_vector_store(uploaded_files)
            st.session_state.messages.append({"role": "assistant", "content": "Documents processed. I'm ready to answer questions about them!"})
    
    st.markdown("---")
    if st.button("Reset Conversation"):
        reset_conversation()
        st.rerun()

# Load the default knowledge base
default_vector_store = load_default_vector_store()

# --- Interactive Chat ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about NHAI..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Determine which retriever to use
    retriever_to_use = None
    search_source = ""
    
    if st.session_state.user_vector_store:
        retriever_to_use = st.session_state.user_vector_store.as_retriever()
        search_source = "your uploaded documents"
    elif default_vector_store:
        retriever_to_use = default_vector_store.as_retriever()
        search_source = "the general NHAI knowledge base"
    
    # Get and display assistant response
    if retriever_to_use:
        with st.chat_message("assistant"):
            with st.spinner(f"Searching in {search_source}..."):
                thinking_process, final_answer, context = get_response_and_context(retriever_to_use, prompt)
                
                # Clean up the thinking process text to avoid f-string errors with backslashes
                cleaned_thinking_process = thinking_process.replace("\n", " ")

                # Format the response with "Thinking" and "Final Answer" sections
                full_response = f"""
### Thinking Process
> *{cleaned_thinking_process}*

***
### Final Response
{final_answer}
"""
                st.markdown(full_response)
                with st.expander("Show Retrieved Context"):
                    st.info(context)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.error("No knowledge base is available. Please run the build script or upload documents.")
        st.session_state.messages.append({"role": "assistant", "content": "I apologize, but no knowledge base is available for me to search."})