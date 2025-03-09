import streamlit as st
import logging
import os
import shutil
import urllib.parse
import pdfplumber
import ollama
from PyPDF2 import PdfReader
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Any
from dotenv import load_dotenv
import openparse
import os
import glob

# Load environment variables from .env file
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
)


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=True)
def extract_model_names() -> Tuple[str, ...]:
    """Extracts available model names from Ollama API with proper structure handling."""
    logger.info("Extracting model names")
    try:
        models_info = ollama.list()

        # Ensure response is valid
        if not models_info or not hasattr(models_info, "models") or not models_info.models:
            logger.error("No models found in Ollama API response.")
            return ()

        # Extract model names properly
        model_names = tuple(
            model.model for model in models_info.models if model.model and "llama2" not in model.model
        )

        if not model_names:
            logger.error("No valid models found in response.")
            return ()

        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return ()


@st.cache_resource
def get_embeddings():
    logger.info("Using Ollama Embeddings")
    return OllamaEmbeddings(model="nomic-embed-text")

def get_text_chunks(pdf_path):
    try:
        basic_doc_path = pdf_path
        parser = openparse.DocumentParser()
        parsed_basic_doc = parser.parse(basic_doc_path)    
        chunks, metadata = [], []    
        for nodes in parsed_basic_doc.nodes:
            chunks.append(nodes.text)
            metadata.append({"node": nodes})
        return chunks, metadata
            
    except Exception as e:
        logger.error(f"Error in get_text_chunks function: {e}")
        return [], []


def get_vector_store(text_chunks, metadata):
    vector_store = None
    try:
        embeddings = get_embeddings()
        # Add metadata to each chunk
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
        print(vector_store) 
        vector_store.save_local("faiss_index")
    except Exception as e:
        logger.error(f"Error at get_vector_store function: {e}")
    return vector_store

@st.cache_resource
def get_llm(selected_model: str):
    logger.info(f"Getting LLM: {selected_model}")
    return ChatOllama(model=selected_model, temperature=0.1)


def process_question(question: str, vector_db: FAISS, selected_model: str) -> str:
    logger.info(f"Processing question: {question} using model: {selected_model}")
    llm = get_llm(selected_model)

    # Define the query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="Original question: {question}",
    )


    # # Create retriever with LLM for multiple query retrieval
    # retriever = MultiQueryRetriever.from_llm(
    #     vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    # )

    # Retrieve only the best document (k=1)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    # Define the answer template
    template = (
        "Answer the question as detailed as possible from the provided context only. \n"
        "Do not generate a factual answer if the information is not available. \n"
        "If you do not know the answer, respond with \"I don’t know the answer as not sufficient information is provided in the PDF.\"\n"
        "Context:\n {context}?\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )

    prompt = ChatPromptTemplate.from_template(template)

    # Set up the chain with retriever and LLM
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Get the response from the chain
    langchain.debug = True
    response = chain.invoke(question)

    # Check if the retrieved context is relevant or not
    if "I don’t know the answer" in response or not response.strip():
        return "I don’t know the answer as not sufficient information is provided in the PDF."

    # Retrieve metadata from the vector database's relevant documents.
    # This assumes that the underlying retriever returns Document objects with a 'metadata' attribute.
    docs = vector_db.as_retriever(search_kwargs={"k": 1}).get_relevant_documents(question)
    metadata_list = [doc.metadata for doc in docs if hasattr(doc, "metadata") and doc.metadata]
    logger.info("Question processed and response generated with metadata")
    return response, metadata_list


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    logger.info(
        f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages



def delete_vector_db() -> None:
    logger.info("Deleting vector DB")
    st.session_state.pop("pdf_pages", None)
    st.session_state.pop("file_upload", None)
    st.session_state.pop("vector_db", None)
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
        logger.info("FAISS index deleted")
    st.success("Collection and temporary files deleted successfully.")
    logger.info("Vector DB and related session state cleared")
    st.rerun()

def save_uploaded_file(uploaded_file):
    """Save uploaded PDF to a temporary file and return its path."""
    file_path = f"./Data/{uploaded_file.name}"
    files = glob.glob(file_path)
    for f in files:
        os.remove(f)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    st.title("🧠 Ollama Chat with PDF RAG", anchor=False)

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.9rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    available_models = extract_model_names()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    selected_model = st.sidebar.selectbox(
        "Pick a model available locally on your system", available_models) if available_models else ""
    pdf_docs = st.sidebar.file_uploader(
        "Upload PDFs", accept_multiple_files=True)
    
    for pdf in pdf_docs:
        pdf_path = save_uploaded_file(pdf)
    

    if st.sidebar.button("Process PDF") and pdf_docs:
        with st.spinner("Processing..."):
            # text, page_mapping = get_pdf_text(pdf_docs)
            chunks, metadata = get_text_chunks(pdf_path)
            st.session_state["vector_db"] = get_vector_store(chunks, metadata)
            st.success("Done")
        pdf_pages = extract_all_pages_as_images(
            pdf_docs[0])  # Assuming single file upload
        st.session_state["pdf_pages"] = pdf_pages
		
    if st.sidebar.button('⚠️ Delete collection'):
        delete_vector_db()

    message_container = st.container(height=500, border=True)
    for message in st.session_state["messages"]:
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter a prompt here..."):
        try:
            st.session_state["messages"].append(
                {"role": "user", "content": prompt})
            message_container.chat_message("user", avatar="😎").markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner(":green[processing...]"):
                    if st.session_state["vector_db"] is not None:
                        response, metadata = process_question(
                            prompt, st.session_state["vector_db"], selected_model
                        )
                        st.markdown(response)
                        n = [i['node'] for i in metadata]
                        # Display metadata with a button
                        if metadata:
                            pdf = openparse.Pdf('/Users/razim/Downloads/lat/Data/2501.12948v1.pdf')
                            pdf.export_with_bboxes(
                                n,
                                output_pdf="/Users/razim/Downloads/lat/meta/marked-up.pdf"
                            )
                  

                        st.markdown("### Sources:")
                        for i, meta in enumerate(n, start=1):
                            pdf_path = f"/Users/razim/Downloads/lat/meta/marked-up.pdf"  
                            page_number = meta.bbox[0].page  # Page number
                            print(pdf_path)
                            # Create the URL to open `op.py`
                            pdf_viewer_url = f"http://localhost:8502/?file={pdf_path}&page={page_number+1}"

                            # Use `st.markdown` with proper HTML formatting
                            button_html = f"""
                            <a href="{pdf_viewer_url}" target="_blank">
                                <button style="margin:5px; padding:10px; font-size:16px;">
                                    📖 Source {i})
                                </button>
                            </a>
                            """
                            st.markdown(button_html, unsafe_allow_html=True)



                        
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": response}
                        )
                    else:
                        response = "Please upload and process a PDF file first."
                        st.warning(response)

        except Exception as e:
            st.error(e, icon="⚠️")
            logger.error(f"Error processing prompt: {e}")








if __name__ == "__main__":
    main()