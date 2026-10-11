from src.helper import embedding, text_splitter, load_repo
from langchain_chroma import Chroma



docs=load_repo("repo/")
texts=text_splitter(docs)
embeddings=embedding()

vecstore=Chroma.from_documents(texts,embedding=embeddings,persist_directory="DB")

