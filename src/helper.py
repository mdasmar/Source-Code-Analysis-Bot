from git import Repo
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def repo_clone(repo_url):
    os.makedirs("repo",exist_ok=True)
    repo_path="repo/"
    Repo.clone_from(repo_url,to_path=repo_path)

def load_repo(repo_path):
    loader=GenericLoader.from_filesystem(repo_path,
                glob="**/*",
                suffixes=[".py"],
                parser=LanguageParser(language=Language.PYTHON,parser_threshold=500))
    docs=loader.load()
    return docs

def text_splitter(docs):
    docs_splitter=RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                chunk_size=500,
                chunk_overlap=20)
    texts=docs_splitter.split_documents(docs)
    return texts

def embedding():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings