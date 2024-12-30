from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain

def extract_and_vectordb(pdf_file):
    reader = PdfReader(pdf_file)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    
    text_splitter = RecursiveCharacterTextSplitter()
    texts = text_splitter.split_text(raw_text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vec_db = FAISS.from_texts(texts, embeddings)
    
    return vec_db

template = """
You are a helpful assistant who provides accurate and straightforward answers based on the given document.\n
Given the following document:\n{context}\n\nAnswer the question:\n{question}\n\n
Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

from langchain.memory import ConversationBufferMemory 

def initialize_chain(vector_db, model_name, api_key, temperature):
    llm = HuggingFaceHub(
        repo_id=model_name,
        model_kwargs={"temperature": temperature, "max_tokens": 512},
        huggingfacehub_api_token=api_key,
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        #memory=memory
        )
    return conversation_chain

def postprocess(response):
    query_index = response.rfind("Answer:")
    if query_index != -1:
        answer = response[query_index + len("Answer:"):].strip()
        answer = answer.lstrip("\n\"")
        return answer
    else:
        return ""

def get_the_answer(conversation_chain, question, chat_history):
    conv_response = conversation_chain.invoke({"question": question, "chat_history": chat_history})
    response = postprocess(conv_response['answer'])
    return response

# chat_history = []

import streamlit as st

def main():
    st.title("💬 PDF Chatbot")

    # Sidebar controls
    model_name = st.sidebar.selectbox(
        "Choose the Model", 
        ["mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.2-3B-Instruct"], 
        index=0
    )
    api_key = st.sidebar.text_input("Enter HuggingFace API Key:", type="password")
    temperature = st.sidebar.slider("Set Temperature", min_value=0.05, max_value=1.0, value=0.7, step=0.05)
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

    # Initialize session state variables
    if 'vec_db' not in st.session_state:
        st.session_state.vec_db = None
    if 'chat_display' not in st.session_state:
        st.session_state.chat_display = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [] # Not appending anything to

    # Buttons
    if st.sidebar.button("Read PDF") and uploaded_file:
        with st.spinner("Processing PDF..."):
            try:
                st.session_state.vec_db = extract_and_vectordb(uploaded_file)
                st.sidebar.success("PDF successfully processed!")
            except Exception as e:
                st.sidebar.error(f"Error processing PDF: {e}")

    if st.sidebar.button("Delete PDF"):
        st.session_state.vec_db = None
        st.sidebar.info("PDF deleted and vector database cleared.")

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_display = []
        st.session_state.chat_history = []

    user_input = st.chat_input("Enter your message:")

    if user_input:
        st.session_state.chat_display.append(("user", user_input))

        for sender, message in st.session_state.chat_display:
            st.chat_message(sender).write(message)

        if st.session_state.vec_db:
            with st.spinner("Thinking..."):
                try:
                    conversation_chain = initialize_chain(
                        vector_db=st.session_state.vec_db,
                        model_name=model_name,
                        api_key=api_key,
                        temperature=temperature,
                    )
                    response = get_the_answer(conversation_chain, user_input, st.session_state.chat_history)
                    st.session_state.chat_display.append(("assistant", response))

                    st.chat_message("assistant").write(response)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        else:
            st.error("Please upload and process a PDF first.")

if __name__ == "__main__":
    main()


