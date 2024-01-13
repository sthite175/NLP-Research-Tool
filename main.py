import os
import numpy as np

from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chains import SimpleSequentialChain, SequentialChain

from langchain.vectorstores import Pinecone, FAISS
import pinecone

from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.vectorstores import Pinecone,FAISS
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st
import time
load_dotenv()


llm = OpenAI(temperature=0.9, max_tokens=500)

# Set the title of the app
st.title("NLP Research Tool")

st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


process_urls_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

filepath = "faiss_store"
embedding = OpenAIEmbeddings()
if process_urls_clicked:
    ### LOAD DATA
    loader = UnstructuredURLLoader(urls=urls)
    #main_placeholder = st.empty()
    main_placeholder.text("Data Loading.....Started")
    data = loader.load()

    ### SPLIT DATA
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n" ,".", ","],
        chunk_size=500, chunk_overlap=10
     )
    
    main_placeholder.text("Data Loading.....Started")
    docs = text_splitter.split_documents(data)

    ### EMBEDDING AND SAVING DATA TO FAISS INDEX
    vectorindex_openai = FAISS.from_documents(docs, embedding)
    main_placeholder.text("Data Loading.....Started")
    time.sleep(2)

    # SAVE  FILE
    vectorindex_openai.save_local(filepath)


query = main_placeholder.text_input("Questions: ")
#query = st.text_input("Question:")
if query:
    if os.path.exists(filepath):
        # LOAD Folder
        vector_file = FAISS.load_local(filepath, embedding)#OpenAIEmbeddings())
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_file.as_retriever())

        result = chain({"question": query},return_only_outputs=True)
        # Display The Answer
        st.subheader("Answer:")
        st.write(result['answer'])

        # Display sources if available 
        sources = result.get('sources',"")
        if sources:
            st.subheader("Sources")
            sources_list = sources.split("\n") # Split the sources by newline
            for source in sources_list:
                st.write(source)

