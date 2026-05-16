from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"
DATA_PATH = "data/sample_docs"


def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return embeddings


def load_and_split_documents():
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vector_store():
    chunks = load_and_split_documents()

    if not chunks:
        print("No documents found in data/sample_docs. Add .txt files first.")
        return None

    embedding_fn = get_embedding_function()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=CHROMA_PATH
    )

    print(f"Vector store built with {len(chunks)} chunks.")
    return vector_store


def load_vector_store():
    embedding_fn = get_embedding_function()

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_fn
    )
    return vector_store