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
import json
from langchain.chains.question_answering import load_qa_chain
from image_links import image_links
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

load_dotenv()
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
            width: 25vw;  /* adjust this value as needed */
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

import json

# ---------- Load Question Papers JSON ----------
with open("papers.json", "r") as f:
    papers_data = json.load(f)

# ---------- Question Paper Matching Logic ----------
def get_question_paper_link(prompt):
    with open("papers.json") as f:
        data = json.load(f)

    prompt_lower = prompt.lower()

    # Flexible regex patterns
    course_match = re.search(r"\b(mca|b\.?tech)\b", prompt_lower)
    semester_match = re.search(r"\bsem(?:ester)?[ -]?(\d)\b", prompt_lower)
    year_match = re.search(r"\b(20\d{2})\b", prompt_lower)
    exam_match = re.search(r"\b(midsem|endsem)\b", prompt_lower)

    # Clean extracted values
    course = course_match.group(1).replace(".", "").upper() if course_match else None
    semester = int(semester_match.group(1)) if semester_match else None
    year = year_match.group(1) if year_match else None
    exam = exam_match.group(1).lower() if exam_match else None

    subject = None
    subject_found = None

    # Dynamically find subject from JSON
    for course_entry in data:
        for subject_entry in course_entry["subjects"]:
            subject_name = subject_entry["subject"].lower()
            aliases = [subject_name] + subject_entry.get("aliases", [])
            
            if any(alias in prompt_lower for alias in aliases):
                subject_found = subject_entry
                matched_course = course_entry["course"]
                break
        if subject_found:
            break
    # for course_entry in data:
    #     for subject_entry in course_entry["subjects"]:
    #         subject_name = subject_entry["subject"].lower()
    #         if subject_name in prompt_lower:
    #             subject_found = subject_entry["subject"]
    #             break
    #     if subject_found:
    #         break

    # for All papers for a semester
    if course and semester and "paper" in prompt_lower and ("all" in prompt_lower or not subject_found):
        results = []
        for course_entry in data:
            if course_entry["course"].lower() == course.lower():
                for subject_entry in course_entry["subjects"]:
                    if subject_entry.get("semester") == semester:
                        subject_name = subject_entry["subject"]
                        papers = subject_entry.get("papers", {})
                        for exam_type, years in papers.items():
                            for y, link in years.items():
                                results.append(f"**{subject_name}** - {exam_type.capitalize()} {y} → [View Paper]({link})")
        # return "\n".join(results) if results else None
        return "\n\n".join(results) if results else None


    # for Specific subject, exam, year
    if course and subject_found and exam and year:
        for course_entry in data:
            if course_entry["course"].lower() == course.lower():
                for subject_entry in course_entry["subjects"]:
                    # if subject_entry["subject"].lower() == subject_found.lower():
                    if subject_entry["subject"].lower() == subject_found["subject"].lower():
                        try:
                            return subject_entry["papers"][exam][year]
                        except KeyError:
                            return None

    return None

# ---------- Keyword Matcher for Paper-related Queries ----------
def is_question_paper_prompt(prompt):
    keywords = ["paper", "question paper", "midsem", "endsem", "exam"]
    return any(word in prompt.lower() for word in keywords)

# -------------------- Prompt Template --------------------

def custom_prompt():
    
    template = """
    You are a chatbot built to answer queries ONLY using the provided academic dataset from MNNIT. You are to respond only with information present in the documents, including exam papers, notes, and questions, without restriction.
    
    Answer the question based on the context below. If the answer cannot be determined from the context, stop by saying "I don't know".

    If the question is about an algorithm, provide:
    A step-by-step explanation
    Pseudocode

    Context:
    {context}

    Question:
    {question}
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])


# -------------------- Language Codes --------------------
LANGUAGES = {
    "English": "en_XX",
    "Hindi": "hi_IN",
    "Punjabi": "pa_IN",
    "Gujarati": "gu_IN",  
    "Bengali": "bn_IN",   
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
    encoded = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=128,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        length_penalty=1.0,
        do_sample=False
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def safe_translate(text, src_lang, tgt_lang, tokenizer, model):
    return translate(text, src_lang, tgt_lang, tokenizer, model)
  

@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    return Together(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        together_api_key=os.environ["TOGETHER_API_KEY"],
        temperature=0.7,
        max_tokens=8192, 
        # max_tokens=1024,
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

# ---------- Clean & Truncate OCR/Prompt Input ----------
def preprocess_text(text, max_chars=500):
    cleaned = text.strip().replace("\n", " ")
    return cleaned[:max_chars]

# ---------- Format Output for Readability ----------
def clean_response(text):
    import re
    text = re.sub(r"\s*(\d+\.\s)", r"\n\1", text)
    
    # to match lines starting with 'function', 'def', 'for', 'if', etc.
    if re.search(r"\b(function|def|for|if|while|return)\b", text):
        #  to find block starting with function and ending with 'return ...; }' or similar
        match = re.search(r"(function.*?})", text, re.DOTALL)
        if match:
            code_block = match.group(1)
            text = text.replace(code_block, f"```pseudo\n{code_block}\n```")
    
    return text.strip()

# def extract_metadata_from_prompt(prompt: str):
#     prompt = prompt.lower()

#     # Extract year
#     year_match = re.search(r"\b(20\d{2})\b", prompt)
#     year = year_match.group(1) if year_match else None

#     # Extract exam type
#     if "mid" in prompt:
#         exam = "midsem"
#     elif "end" in prompt:
#         exam = "endsem"
#     else:
#         exam = None

#     # Extract subject
#     subject_match = re.search(r"give me (\w+)\s+paper", prompt)
#     subject = subject_match.group(1) if subject_match else None

#     return {
#         "subject": subject,
#         "year": year,
#         "exam": exam
#     }

# ---------- Get Image if Topic Found ----------
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
    # st.sidebar.header("🌐 Language Preferences")
    # target_language = st.sidebar.selectbox(
    #     "Chat history to be displayed here",
    #     options=["English", "Hindi", "Spanish"],
    #     index=0
    # )

    st.markdown('<div class="big-title">A New Multilingual Chatbot using Machine Learning</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # selected_language = st.selectbox("🌍 Choose Target Language", list(LANGUAGES.keys()))
    selected_language = "English"
    

    translation_model, tokenizer = load_translation_model()
    
    vectorstore = load_vectorstore()
    llm = load_llm()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f'<div class="chatbox">{msg["content"]}</div>', unsafe_allow_html=True)

    
    
    prompt = st.chat_input("Ask something...")
    
   
    if prompt:

        src_lang = LANGUAGES[selected_language]
        tgt_lang = "en_XX"

        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Initialize retriever without any filter
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
            chain = load_qa_chain(llm, chain_type="stuff", prompt=custom_prompt())
            result = chain.run(
                input_documents=meaningful_docs,
                question=prompt,
            )

            translated_response = safe_translate(result, "en_XX", LANGUAGES[selected_language], tokenizer, translation_model)
            
            final_response = translated_response if len(translated_response.split()) > 5 else result
            final_response = clean_response(final_response)
                        
            final_response = str(final_response)
            # Check if any relevant image exists for the algorithm or topic in the prompt
            image_markdown = get_image_from_prompt(prompt)
            if image_markdown:
                final_response += f"\n\n{image_markdown}"
            
            # If it’s a question paper request, override response
            if is_question_paper_prompt(prompt):
                paper_link = get_question_paper_link(prompt)
                if paper_link:
                    # if paper_link.startswith("**"):  # It's a list of formatted markdown entries
                    if "\n" in paper_link:
                        final_response = paper_link
                    else:  # It's a single direct link
                        final_response = f"📄 [Click here to view the question paper]({paper_link})"
                else:
                    # Only show LLM response if no link was found
                    uncertainty_phrases = ["i don't know", "do not have access", "cannot find", "i'm not sure"]
                    if any(phrase in final_response.lower() for phrase in uncertainty_phrases):
                        final_response = "Sorry, I couldn't find a matching paper link."
            else:
                # For non-paper queries, check for fallback phrases
                uncertainty_phrases = ["i don't know", "do not have access", "cannot find", "i'm not sure"]
                if any(phrase in final_response.lower() for phrase in uncertainty_phrases):
                    final_response = "Sorry, I couldn't find anything relevant in the documents you uploaded to answer that."

            # if is_question_paper_prompt(prompt):
            #     paper_link = get_question_paper_link(prompt)
            #     if paper_link:
            #         final_response += f"Here is the question paper link you asked for:\n\n[Click to View Paper]({paper_link})"


            st.chat_message("assistant").markdown(final_response)
            # st.chat_message("assistant").markdown(
            #     f"<div style='white-space: pre-wrap; word-break: break-word;'>{final_response}</div>",
            #     unsafe_allow_html=True
            # )
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response
            })
    
    uploaded_image = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        image = image.resize((image.width // 2, image.height // 2))  # Resize by half
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
                        
                st.session_state.image_uploader = None
                st.session_state.extracted_text = ""
        else:
            st.warning("⚠️ Could not extract readable text from the image.")

if __name__ == "__main__":
    main()
