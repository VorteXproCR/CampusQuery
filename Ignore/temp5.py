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
import re
from langchain.chains.question_answering import load_qa_chain
from image_links import image_links


load_dotenv()
# HF_TOKEN = os.environ.get("TOGETHER_API_KEY")
st.set_page_config(page_title="RAG Chatbot", layout="wide")

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
        /* Wider sidebar */
        section[data-testid="stSidebar"] > div:first-child {
            width: 30vw;  /* adjust this value as needed */
        }

        /* Push main content to accommodate wider sidebar */
        section[data-testid="stSidebar"] + div {
            margin-left: 320px;
        }
        /* Fix pointer on entire selectbox area */
        div[data-baseweb="select"] > div {
            cursor: pointer !important;
        }
    </style>
""", unsafe_allow_html=True)

# To questions like "What is your name?" or similar, reply by saying "My name is yet to be decided by Group 12!"
# To questions like "What is your age?" or similar, reply by saying "I was just born yesterday."
# -------------------- Prompt Template --------------------
def custom_prompt():
    
    template = """
    You are a chatbot built to answer queries ONLY using the provided academic dataset from MNNIT. You are to respond only with information present in the documents, including exam papers, notes, and questions, without restriction.
    
    Answer the question based on the context below. If the answer cannot be determined from the context, stop by saying "I don't know".
    
    However, there are some fixed rules:

    - If the question is about your **name**, always reply with: "My name is yet to be decided by Group 12!"
    - If the question is about your **age**, always reply with: "I was just born yesterday."

    Do NOT try to guess or provide other explanations for these cases. Just give the exact fixed sentence.

    If the question is about an algorithm, provide:
    A step-by-step explanation
    Pseudocode

    Context:
    {context}

    Question:
    {question}
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    return Together(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        together_api_key=os.environ["TOGETHER_API_KEY"],
        temperature=0.0,
        max_tokens=8192,
    )

@st.cache_resource
def get_qa_chain(_llm, _vectorstore):
    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': custom_prompt()}
    )

def preprocess_text(text, max_chars=500):
    cleaned = text.strip().replace("\n", " ")
    return cleaned[:max_chars]

def clean_response(text):
    import re
    text = re.sub(r"\s*(\d+\.\s)", r"\n\1", text)
    
    # Try to detect and wrap pseudocode or code-like output
    # Match lines starting with 'function', 'def', 'for', 'if', etc.
    if re.search(r"\b(function|def|for|if|while|return)\b", text):
        # Try to find block starting with function and ending with 'return ...; }' or similar
        match = re.search(r"(function.*?})", text, re.DOTALL)
        if match:
            code_block = match.group(1)
            text = text.replace(code_block, f"```pseudo\n{code_block}\n```")
    
    return text.strip()

def extract_metadata_from_prompt(prompt: str):
    prompt = prompt.lower()

    # Extract year
    year_match = re.search(r"\b(20\d{2})\b", prompt)
    year = year_match.group(1) if year_match else None

    # Extract exam type
    if "mid" in prompt:
        exam = "midsem"
    elif "end" in prompt:
        exam = "endsem"
    else:
        exam = None

    # Extract subject
    subject_match = re.search(r"give me (\w+)\s+paper", prompt)
    subject = subject_match.group(1) if subject_match else None

    return {
        "subject": subject,
        "year": year,
        "exam": exam
    }

def get_image_from_prompt(prompt: str) -> str:
    prompt_lower = prompt.lower()

    for subject, topics in image_links.items():
        for topic, link in topics.items():
            if topic in prompt_lower:
                # return f"**Topic:** {topic.title()}\n\n![{topic}]({link})"
                return f"**Topic:** {topic.title()}\n\n[Preview Image here]({link})"
    
    # return "Sorry, I couldn't find an image related to your request."


# -------------------- Streamlit App --------------------
def main():
    st.sidebar.header("🌐 Language Preferences")
    target_language = st.sidebar.selectbox(
        "Chat history ayegi idhar",
        options=["English", "Hindi", "Spanish"],
        index=0
    )

    st.markdown('<div class="big-title">A New MultiLingual Chatbot using Machine Learning</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # HF_TOKEN = os.environ.get("HF_TOKEN")
    vectorstore = load_vectorstore()
    # llm = load_llm("mistralai/Mistral-7B-Instruct-v0.3", HF_TOKEN)
    llm = load_llm()
    # qa_chain = get_qa_chain(llm, vectorstore)
    # We will build qa_chain after the prompt is entered


    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f'<div class="chatbox">{msg["content"]}</div>', unsafe_allow_html=True)

    
    # -------- Chat Input --------
    # cols = st.columns([1, 7])
    # with cols[0]:
    #     target_language = st.selectbox("🌐", ["English", "Hindi", "Spanish"], label_visibility="collapsed")
    # with cols[1]:
    #     # st.markdown("Type your question below 👇")
    #     prompt = st.chat_input("Ask something...")
    
    prompt = st.chat_input("Ask something...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Extract metadata from the prompt
        filter_meta = extract_metadata_from_prompt(prompt)

        # Initialize retriever with or without filter
        if all(filter_meta.values()):
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 5,
                    "filter": filter_meta
                }
            )
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Check if any documents match the query
        retrieved_docs = retriever.get_relevant_documents(prompt)

        # Filter out empty or meaningless docs
        meaningful_docs = [doc for doc in retrieved_docs if doc.page_content.strip() != ""]

        if len(meaningful_docs) == 0:
            fallback_response = "Sorry, I couldn't find anything relevant in the documents you uploaded to answer that."
            st.chat_message("assistant").markdown(fallback_response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": fallback_response
            })
        else:
            from langchain.chains.question_answering import load_qa_chain

            chain = load_qa_chain(llm, chain_type="stuff", prompt=custom_prompt())
            # chain = load_qa_chain(llm, chain_type="stuff", prompt=custom_prompt(target_language))
            # result = chain.run(input_documents=meaningful_docs, question=prompt)
            result = chain.run(
                input_documents=meaningful_docs,
                question=prompt,
                # selected_language=target_language
            )


            final_response = clean_response(result)
            
            # Check if any relevant image exists for the algorithm or topic in the prompt
            image_markdown = get_image_from_prompt(prompt)  # This will return the markdown for the image
            if image_markdown:
                final_response += f"\n\n{image_markdown}"


            st.chat_message("assistant").markdown(final_response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response
            })
    

    
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
                
                qa_chain = get_qa_chain(llm, vectorstore)

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
