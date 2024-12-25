import os
import shutil
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_chroma import Chroma
from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Base directory for temporary vectorstores
VECTORSTORE_DIR = "./vectorstores"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


import logging

logging.basicConfig(level=logging.INFO)

# Directory for storing vectorstore
VECTORSTORE_DIR = "./vectorstores"

def fetch_and_process_url(url: str, session_id: str):
    try:
        # Fetch data from the URL
        response = requests.get(url)
        response.raise_for_status()
        raw_text = response.text
        logging.info("Fetched raw text successfully.")
    except Exception as e:
        logging.error(f"Failed to fetch data from URL: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch data from URL: {e}")

    # Split text into chunks 
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],  # Split by larger, then smaller units
            chunk_size=3000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)
        logging.info(f"Split text into {len(texts)} chunks.")
    except Exception as e:
        logging.error(f"Error while splitting text: {e}")
        raise HTTPException(status_code=500, detail=f"Error while splitting text: {e}")

    # Warn about oversized chunks
    for i, chunk in enumerate(texts):
        if len(chunk) > 3000:
            logging.warning(f"Chunk {i} exceeds the maximum size of 3000 characters.")

    # Initialize vectorstore
    try:
        user_vectorstore_dir = os.path.join(VECTORSTORE_DIR, session_id)
        if os.path.exists(user_vectorstore_dir):
            shutil.rmtree(user_vectorstore_dir)  # Clear previous vectorstore
            logging.info(f"Cleared existing vectorstore for session: {session_id}")

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        metadatas = [{"source": "unknown"} for _ in texts]

        vectorstore = Chroma(
            collection_name="user_documents",
            embedding_function=embeddings,
            persist_directory=user_vectorstore_dir,
        )
        
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas
        )
        logging.info(f"Vectorstore created for session: {session_id}")
        
    except Exception as e:
        logging.error(f"Error initializing vectorstore: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error initializing vectorstore: {e}"
        )

    return vectorstore.as_retriever()


model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

from langchain_aws import ChatBedrock

llm= ChatBedrock(
        model_id=model_id,
        model_kwargs=dict(temperature=0),
    )

def initialize_qa_chain(retriever):
    prompt_template = """
    Use the following piece of information to answer the user's question.
    If you don't know the answer, just say that you don't know. Do not make up an answer. Just say the answer you dont need to say anything except the answer like "based on..." just answer only.
    Context: {context}
    Question: {question}
    
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=False, chain_type_kwargs={"prompt": prompt}
    )

import time
# Add the cleanup function
def cleanup_old_sessions(max_age_hours=1):
    """Remove vectorstores older than specified hours"""
    try:
        for session_dir in os.listdir(VECTORSTORE_DIR):
            session_path = os.path.join(VECTORSTORE_DIR, session_dir)
            # Check directory age
            if time.time() - os.path.getmtime(session_path) > max_age_hours * 3600:
                shutil.rmtree(session_path)
                logging.info(f"Cleaned up old session: {session_dir}")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
