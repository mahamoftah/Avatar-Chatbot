import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.api_core.client_options import ClientOptions
from PyPDF2 import PdfReader
import os
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class Gemini:

    def __init__(self) -> None:
        self.messages = []
        # add google_key to .env file
        load_dotenv(find_dotenv(), override=True)
        os.environ['PINECONE_API_KEY'] = 'c8519fb3-b5b7-461e-afa7-0090e4a5fa43'
        self.vec_db = self.load_vec_db()         # load pinecone database (not free because of using openai embedding model), it was better than faiss when retrieving arabic answers. so we don't have to use it now.
        self.vector_db = self.load_vector_db()   # load faiss database (free), should download faiss_index folder

    def load_vec_db(self):
        return Pinecone.from_existing_index("gp-iti", OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536))

    def load_vector_db(self):
        return FAISS.load_local("utils/Model/faiss_index", self.embedding_google(), allow_dangerous_deserialization=True)

    def embedding_google(self):
        return GoogleGenerativeAIEmbeddings(model='models/embedding-001', task_type="SEMANTIC_SIMILARITY")

    def search_similar_context(self, vector_db, question, n):
        if vector_db:
            docs = vector_db.similarity_search(question, k=n)
            return docs


    def process_question(self, question):
        return genai.GenerativeModel('gemini-1.5-flash').generate_content(question, stream=False).text


    def generate(self, question, lang = 'en'):

        similar_text = "You are a Multi Task AI Agent"
        self.messages.append({"role": "user", "content": question})

        similar_context = self.search_similar_context(self.vector_db, question, 5)
        for context in similar_context:
            similar_text += context.page_content

        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages])
        combined_input = f"{conversation_history}\nuser: {question}\nAI:"
        combined_input += similar_text

        response = self.process_question(combined_input)
        self.messages.append({"role": "AI", "content": response})

        return response

# Testing
# 
# from Model.Gemini import *
# gemini_ = Gemini()
# answer = gemini_.gemini(question)

