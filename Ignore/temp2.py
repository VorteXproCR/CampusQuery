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
from langchain_community.llms import Together


load_dotenv()
# HF_TOKEN = os.environ.get("TOGETHER_API_KEY")
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# import cv2
# import numpy as np

# def preprocess_image_for_ocr(pil_image):
#     # Convert to grayscale
#     image = np.array(pil_image.convert("L"))

#     # Apply binary thresholding
#     _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     return Image.fromarray(image)


# -------- Styling --------
st.markdown("""
    <style>
        .big-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .subtle {
            color: #888;
            font-size: 0.9rem;
        }
        .chatbox {
            background-color: #f0f4f8;
            border-style: solid;
            border-radius: 10px;
            padding: 10px 15px;
            margin: 5px 0;
        }
        .upload-box-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            border: 2px dashed #aaa;
            border-radius: 20px;
            padding: 40px 30px;
            background: linear-gradient(145deg, #ffffff, #f0f0f0);
            box-shadow: 8px 8px 20px #d1d1d1, -8px -8px 20px #ffffff;
            transition: 0.3s ease-in-out;
            margin-bottom: 1rem;
            cursor: pointer;
        }
        .upload-box-wrapper:hover {
            border-color: #6c63ff;
            background-color: #fafaff;
            box-shadow: 0 0 12px rgba(108, 99, 255, 0.3);
        }
        .upload-icon {
            font-size: 48px;
            color: #6c63ff;
            margin-bottom: 10px;
        }
        .file-upload-label {
            font-size: 1.2rem;
            font-weight: 500;
        }
        .stFileUploader > div {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- Prompt Template --------------------
def custom_prompt():
    template = """
    Use the context below to answer the user's question. Be concise.
    Even if the answer is not given in the context, try to answer it based on your knowledge.
    Context: {context}
    Question: {question}
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)

# @st.cache_resource
# def load_llm(_repo_id, _hf_token):
#     return HuggingFaceEndpoint(
#         repo_id=_repo_id,
#         temperature=0.5,
#         model_kwargs={"token": _hf_token, "max_length": 512}
#     )
@st.cache_resource
def load_llm():
    return Together(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        together_api_key=os.environ["TOGETHER_API_KEY"],
        temperature=0.5,
        max_tokens=8192,
    )

@st.cache_resource
def get_qa_chain(_llm, _vectorstore):
    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': custom_prompt()}
    )

def preprocess_text(text, max_chars=500):
    cleaned = text.strip().replace("\n", " ")
    return cleaned[:max_chars]

def clean_response(text):
    import re
    text = re.sub(r"\s*(\d+\.\s)", r"\n\1", text)
    
    # Add a blank line after each numbered answer
    # text = re.sub(r"(\d+\..*?)(?=\n\d+\.\s|$)", r"\1\n", text, flags=re.DOTALL)
    return text.strip()

# -------------------- Streamlit App --------------------
def main():
    st.markdown('<div class="big-title">A New MultiLungual Chatbot using Machine Learning</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # HF_TOKEN = os.environ.get("HF_TOKEN")
    vectorstore = load_vectorstore()
    # llm = load_llm("mistralai/Mistral-7B-Instruct-v0.3", HF_TOKEN)
    llm = load_llm()
    qa_chain = get_qa_chain(llm, vectorstore)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f'<div class="chatbox">{msg["content"]}</div>', unsafe_allow_html=True)

    
    # -------- Chat Input --------
    prompt = st.chat_input("Ask something...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        
        result = qa_chain.run(prompt)
        final_response = clean_response(result)
        
        # st.write(result['source_documents'])

        st.chat_message("assistant").markdown(final_response)
        
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        
    
    uploaded_image = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        # st.image(image, caption="📸 Uploaded Image", use_column_width=True, output_format="PNG")
        st.image(image, caption="📸 Uploaded Image", use_container_width=False, width=700)

        extracted_text = pytesseract.image_to_string(image)
        processed_text = preprocess_text(extracted_text)

        if processed_text:
            with st.expander("🔍 Extracted OCR Text", expanded=True):
                # st.code(processed_text)
                # st.markdown(f"<div style='white-space: pre-wrap'>{processed_text}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='white-space: pre-wrap; word-break: break-word;'>{processed_text}</div>", unsafe_allow_html=True)


            if st.button("🧠 Ask About This Image"):
                st.chat_message("user").markdown(processed_text)
                st.session_state.messages.append({"role": "user", "content": processed_text})

                with st.chat_message("assistant"):
                    with st.spinner("Generating answer..."):
                        result = qa_chain.run(processed_text)
                        final_response = clean_response(result)
                        st.markdown(final_response)
                        st.session_state.messages.append({"role": "assistant", "content": final_response})
        else:
            st.warning("⚠️ Could not extract readable text from the image.")


if __name__ == "__main__":
    main()
