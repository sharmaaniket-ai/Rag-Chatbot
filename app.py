import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
import tempfile

api=os.getenv("GROQ_KEY")
hf=os.getenv("HF_KEY")

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
st.title("RAG PDF CHATBOT")
st.divider()
st.write("This is a chatbot that can answer questions about a PDF document using RAG techniques.")

if not hf:
    st.error("Set HF_KEY in your .env file to use Hugging Face embeddings.")
    st.stop()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=hf,
)
prompt_template=ChatPromptTemplate.from_template("You are a helpful assistant that answers questions about a PDF document. Use the following context to answer the question: {context} Question: {question}")
parser=StrOutputParser()
llm=ChatGroq(api_key=api, model="openai/gpt-oss-120b",temperature=0.7)
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        st.success(f"Uploaded: {uploaded_file.name}")

    try:
        # Load the PDF using its local file path
        loader = PyMuPDFLoader(temp_file_path)
        documents = loader.load()
        st.success("PDF loaded successfully!")
        st.write(f"Total pages loaded: {len(documents)}")

    finally:
        # Delete the temporary file after loading
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts=text_splitter.split_documents(documents)
    vectorstore=Chroma.from_documents(texts, embeddings)
    retriever=vectorstore.as_retriever()
    chain={'context': retriever, 'question': RunnablePassthrough()} | prompt_template | llm | parser
    question=st.text_input("Ask a question about the PDF document")
    res=chain.invoke(question)
    st.write(res)

