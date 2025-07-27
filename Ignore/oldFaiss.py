# -----------------old working--------------------

# import os
# import re
# import warnings
# from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document

# warnings.filterwarnings("ignore")

# # -------- Step 1: Load Raw PDFs using PyMuPDFLoader (keeps layout better) --------
# DATA_PATH = "data/"

# def extract_metadata_from_filename(filename):
#     name = os.path.splitext(os.path.basename(filename))[0].lower()
#     subject_match = re.search(r"^(.*?)_", name)
#     year_match = re.search(r"(20\d{2})", name)
#     exam = "midsem" if "mid" in name else "endsem" if "end" in name else None

#     return {
#         "subject": subject_match.group(1) if subject_match else None,
#         "year": year_match.group(1) if year_match else None,
#         "exam": exam
#     }

# def load_pdf_files(data_path):
#     loader = DirectoryLoader(
#         data_path,
#         glob="*.pdf",
#         loader_cls=PyMuPDFLoader  # << Changed here
#     )
#     raw_documents = loader.load()

#     enriched_docs = []

#     for doc in raw_documents:
#         metadata = extract_metadata_from_filename(doc.metadata['source'])
#         doc.metadata.update(metadata)
#         enriched_docs.append(doc)

#     return enriched_docs

# documents = load_pdf_files(DATA_PATH)

# # -------- Step 2: Chunking with layout-aware text --------
# def create_chunks(docs):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1200,
#         chunk_overlap=150
#     )
#     chunks = splitter.split_documents(docs)

#     enriched_chunks = []
#     for chunk in chunks:
#         chunk.metadata = chunk.metadata or {}
#         enriched_chunks.append(Document(page_content=chunk.page_content, metadata=chunk.metadata))
#     return enriched_chunks

# text_chunks = create_chunks(documents)

# # -------- Step 3: Generate Embeddings --------
# def get_embedding_model():
#     return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# embedding_model = get_embedding_model()

# # -------- Step 4: Store in FAISS Vector Store --------
# DB_FAISS_PATH = "vectorstore/db_faiss"
# db = FAISS.from_documents(text_chunks, embedding_model)
# db.save_local(DB_FAISS_PATH)
