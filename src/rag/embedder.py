from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import chromadb
import os

load_dotenv()

DATA_PATH = "data/sample_docs"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../../chroma_db")

_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_embedding_function():
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
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
    existing = [c.name for c in _chroma_client.list_collections()]
    if "research_docs" in existing:
        print("Vector store already exists, skipping rebuild.")
        return load_vector_store()

    chunks = load_and_split_documents()
    if not chunks:
        print("No documents found in data/sample_docs.")
        return None

    embedding_fn = get_embedding_function()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        client=_chroma_client,
        collection_name="research_docs"
    )
    print(f"Vector store built with {len(chunks)} chunks.")
    return vector_store


def load_vector_store():
    embedding_fn = get_embedding_function()
    vector_store = Chroma(
        client=_chroma_client,
        collection_name="research_docs",
        embedding_function=embedding_fn
    )
    return vector_store