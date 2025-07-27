import os
import torch
import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv  # ✅ Add this
from langchain_community.llms import Together
load_dotenv()                   # ✅ Load .env variables


# -------------------- Language Codes --------------------
LANGUAGES = {
    "English": "en_XX",
    "Hindi": "hi_IN",
    "Punjabi": "pa_IN",
    "Gujarati": "gu_IN",   # ✅ Added
    "Bengali": "bn_IN",     # ✅ Added
    "Spanish": "es_XX"
    
}

# -------------------- Load Translation Model --------------------
@st.cache_resource
def load_translation_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    return model, tokenizer

# -------------------- Translate --------------------
def translate(text, src_lang, tgt_lang, tokenizer, model):
    if src_lang == tgt_lang:
        return text
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=512
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def safe_translate(text, src_lang, tgt_lang, tokenizer, model):
    return translate(text, src_lang, tgt_lang, tokenizer, model)

# -------------------- Load Vectorstore --------------------
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)
    return db

# -------------------- Load LLM --------------------
def load_llm():
    return Together(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        together_api_key=os.environ["TOGETHER_API_KEY"],
        temperature=0.0,
        max_tokens=8192,
    )
# def load_llm(repo_id, hf_token):
#     return HuggingFaceEndpoint(
#         repo_id=repo_id,
#         temperature=0.5,
#         model_kwargs={"token": hf_token, "max_length": "512"}
#     )

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
    st.set_page_config("Multilingual Chatbot", layout="centered")
    st.title("💬 Multilingual RAG Chatbot")

    # ✅ Dropdown for language selection
    selected_language = st.selectbox("🌍 Choose Your Language", list(LANGUAGES.keys()))

    # Load models
    translation_model, tokenizer = load_translation_model()
    vectorstore = load_vectorstore()

    # Initialize session
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Input box
    prompt = st.chat_input(f"Type your question in {selected_language}")

    if prompt:
        # Translate user input to English (if necessary)
        src_lang = LANGUAGES[selected_language]
        tgt_lang = "en_XX"
        translated_prompt = translate(prompt, src_lang, tgt_lang, tokenizer, translation_model)

        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # QA with vectorstore + LLM
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        print("Token : ", HF_TOKEN);

        qa_chain = RetrievalQA.from_chain_type(
            # llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
            llm = load_llm();
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': custom_prompt()}
        )

        result = qa_chain.run(translated_prompt)

        # Translate the result back to the selected language
        translated_response = safe_translate(result, "en_XX", LANGUAGES[selected_language], tokenizer, translation_model)

        final_response = translated_response if len(translated_response.split()) > 5 else result
        final_response = clean_response(final_response)

        st.chat_message("assistant").markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main()
