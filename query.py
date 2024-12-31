from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from dotenv import load_dotenv
import warnings
import os
warnings.filterwarnings("ignore")

load_dotenv()
llm = ChatOpenAI(api_key=st.secrets["openai_api_key"])
# llm = ChatOpenAI(model="gpt-4o", max_tokens=200)
chat_history = []


# historical messages and the latest user question, and reformulates the question if it makes reference to any information in the historical information
contextualize_q_system_prompt = """Given a chat history and the latest user question {input} \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# build the full QA chain
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question {input}. \
based on {context}.
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# indexing
documents = TextLoader("./docs/faq.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
splits = text_splitter.split_documents(documents)
# db = Chroma.from_documents(documents, OpenAIEmbeddings())
db=Chroma(collection_name="faq_collection", embedding_function=OpenAIEmbeddings(api_key=st.secrets["openai_api_key"]), persist_directory="./faq_db")
db.add_documents(documents)  
retriever = db.as_retriever()

# Retrieve chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Retrieve and generate 
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


def generate_response(query):
    """ Generate a response to a user query"""
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain.invoke({
        "chat_history": chat_history,
        "input": query
    })


def query(query):
    """ Query and generate a response"""
    response = generate_response(query)
    chat_history.extend([HumanMessage(content=query), response["answer"]])
    return response