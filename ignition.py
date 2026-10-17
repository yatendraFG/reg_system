
from pathlib import Path
import requests

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Project configuration
DATA_DIR = Path("data/documents")
VECTOR_DIR = "vector_db"

API_URL = "https://jsonplaceholder.typicode.com/posts"


def load_local_documents():
    documents = []

    for file_path in DATA_DIR.iterdir():

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

        elif file_path.suffix.lower() == ".txt":
            text = file_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "source_type": "document"
                        }
                    )
                )

    return documents


def load_api_data():
    documents = []

    try:
        response = requests.get(
            API_URL,
            timeout=30
        )
        response.raise_for_status()

        records = response.json()

        if not isinstance(records, list):
            raise ValueError("API response must be a list")

        for record in records:

            if not isinstance(record, dict):
                continue

            text = (
                f"Title: {record.get('title', '')}\n"
                f"Content: {record.get('body', '')}"
            )

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": (
                                f"{API_URL}/{record.get('id', '')}"
                            ),
                            "source_type": "api",
                            "record_id": str(record.get("id", ""))
                        }
                    )
                )

    except (requests.RequestException, ValueError) as error:
        print(f"API loading error: {error}")

    return documents


def build_knowledge_library():

    print("1. Loading documents...")
    documents = load_local_documents()

    print("2. Loading external API data...")
    api_documents = load_api_data()
    documents.extend(api_documents)

    if not documents:
        raise RuntimeError(
            "No data loaded. Check documents and API connection."
        )

    print(f"Total source records: {len(documents)}")

    # Split data into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    # Convert chunks into vectors
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("3. Creating embeddings and FAISS index...")

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Persist the knowledge library
    vector_db.save_local(VECTOR_DIR)

    print("4. Knowledge library saved successfully!")
    print(f"Source records: {len(documents)}")
    print(f"Text chunks: {len(chunks)}")


if __name__ == "__main__":
    build_knowledge_library()
