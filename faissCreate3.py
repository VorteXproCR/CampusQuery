import os, re, warnings
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

warnings.filterwarnings("ignore")

# -------- Step 1: Load PDFs recursively and enrich with metadata --------
DATA_PATH = "data/"

def extract_metadata_from_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    subject_match = re.search(r"^(.*?)_", name)
    year_match = re.search(r"(20\d{2})", name)
    exam = "midsem" if "mid" in name else "endsem" if "end" in name else None

    return {
        "subject_hint": subject_match.group(1) if subject_match else None,
        "year": year_match.group(1) if year_match else None,
        "exam": exam
    }

def load_all_pdfs(data_path):
    all_docs = []
    for file_path in Path(data_path).rglob("*.pdf"):
        loader = PyMuPDFLoader(str(file_path))
        docs = loader.load()

        for doc in docs:
            # Infer subject from folder name
            relative_path = file_path.relative_to(data_path)
            parts = relative_path.parts  # e.g., ['Algorithms', 'abc.pdf']
            subject = parts[0] if len(parts) > 0 else None

            doc.metadata.update({
                "subject": subject,
                "source": str(file_path)
            })

            # Optionally enrich with filename-derived metadata
            doc.metadata.update(extract_metadata_from_filename(str(file_path)))

            all_docs.append(doc)
    return all_docs

documents = load_all_pdfs(DATA_PATH)

# -------- Step 2: Chunking --------
def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    enriched_chunks = []
    for chunk in chunks:
        chunk.metadata = chunk.metadata or {}
        enriched_chunks.append(Document(page_content=chunk.page_content, metadata=chunk.metadata))
    return enriched_chunks

text_chunks = create_chunks(documents)

# -------- Step 3: Embeddings --------
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# -------- Step 4: Store in FAISS --------
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)