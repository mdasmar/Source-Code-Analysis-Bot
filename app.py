from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from src.helper import embedding, repo_clone
from flask import Flask, jsonify, request, render_template
import subprocess
import sys
import shutil

app=Flask(__name__)

load_dotenv()

GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

embeddings=embedding()

persist_directory="DB"

vecstore=Chroma(embedding_function=embeddings,persist_directory=persist_directory)

llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash")
memory=ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
qa=ConversationalRetrievalChain.from_llm(llm,retriever=vecstore.as_retriever(search_type="mmr",search_kwargs={"k":8}),memory=memory)

@app.route("/",methods=['GET','POST'])
def index():
    return render_template("index.html")

@app.route("/chatbot",methods=['GET','POST'])
def gitrepo():
    if request.method=='POST':
        user_input=request.form['question']
        repo_clone(user_input)
        subprocess.run([sys.executable,"store_index.py"],check=True)

    return jsonify({"response":str(user_input)})

@app.route("/get",methods=['GET','POST'])
def chat():
    msg=request.form['msg']
    input=msg
    print(input)
    if input=="clear":
        shutil.rmtree("repo",ignore_errors=True)

    result=qa(msg)
    print(result['answer'])
    return str(result['answer'])

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)