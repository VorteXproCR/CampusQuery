import os
import torch
import streamlit as st
from PIL import Image
import pytesseract
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()  # Load HF_TOKEN from .env


# -------------------- Load Vectorstore --------------------
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)
    return db

# -------------------- Load LLM --------------------
def load_llm(repo_id, hf_token):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        model_kwargs={"token": hf_token, "max_length": "512"}
    )

# -------------------- Prompt Template --------------------
def custom_prompt():
    template = """
    Use the context below to answer the user's question. Be concise.
    even if the answer is not given in the context try to answer it based on your knowledge
    Context: {context}
    Question: {question}
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

def clean_response(text):
    return text.strip()

# -------------------- Streamlit App --------------------
def main():
    st.set_page_config("RAG Chatbot with Image OCR", layout="centered")
    st.title("💬 RAG Chatbot with Image Upload")

    # Load vector DB
    vectorstore = load_vectorstore()

    # Initialize session
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # -------- Image Upload Section --------
    st.markdown("📷 **Or upload an image with text**")
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        extracted_text = pytesseract.image_to_string(image).strip()

        if extracted_text:
            st.markdown("📝 **Extracted Text:**")
            st.write(extracted_text)

            if st.button("Ask about extracted text"):
                st.chat_message("user").markdown(extracted_text)
                st.session_state.messages.append({"role": "user", "content": extracted_text})

                # RAG
                HF_TOKEN = os.environ.get("HF_TOKEN")
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm("mistralai/Mistral-7B-Instruct-v0.3", HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=False,
                    chain_type_kwargs={'prompt': custom_prompt()}
                )

                result = qa_chain.run(extracted_text)
                final_response = clean_response(result)

                st.chat_message("assistant").markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
        else:
            st.warning("Could not extract any text from the image.")

    # -------- Text Input Section --------
    prompt = st.chat_input("Type your question here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # RAG
        HF_TOKEN = os.environ.get("HF_TOKEN")
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm("mistralai/Mistral-7B-Instruct-v0.3", HF_TOKEN),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': custom_prompt()}
        )

        result = qa_chain.run(prompt)
        final_response = clean_response(result)

        st.chat_message("assistant").markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main()
