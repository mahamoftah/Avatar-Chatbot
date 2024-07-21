import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os


class Gemini:

    def __init__(self) -> None:
        self.messages = []
        GOOGLE_API_KEY="AIzaSyBaz13UTSLEsag18c_rHQ9yFUbX4sx3YYM"
        genai.configure(api_key=GOOGLE_API_KEY)
        os.environ['PINECONE_API_KEY'] = 'c8519fb3-b5b7-461e-afa7-0090e4a5fa43'
        self.vector_db = self.load_vector_db()
        self.chat = genai.GenerativeModel('gemini-1.5-flash').start_chat(history=[])
        print("Test...")

    def load_vector_db(self):
        return FAISS.load_local("faiss_index", self.embedding_google(), allow_dangerous_deserialization=True)

    def embedding_google(self):
        return GoogleGenerativeAIEmbeddings(model='models/embedding-001', task_type="SEMANTIC_SIMILARITY")

    def load_vector_db(self):
        return FAISS.load_local("faiss_index", self.embedding_google(), allow_dangerous_deserialization=True)

    def search_similar_context(self, vector_db, question, n):
        if vector_db:
            docs = vector_db.similarity_search(question, k=n)
            return docs


    def process_question(self, question):
        return genai.GenerativeModel('gemini-1.5-flash').generate_content(question, stream=False).text


    def gemini(self, question):

        similar_text = ""
        similar_context = self.search_similar_context(self.vector_db, question, 5)
        for context in similar_context:
            similar_text += context.page_content

        # i wanted to pass the context as system prompt but unfortunately there is only two roles [user and model].
        # so i thought that it's posible to pass it as a seperated user prompt, or append it to the input question to pass it as cominaed user prompt
        # prompt = f"Use the given below context to answer the user query. {similar_text} Question: {question}?"
        # or self.messages.append({"role": "user", "parts": [similar_text]})

        self.messages.append({"role": "user", "parts": [similar_text]})
        self.messages.append({"role": "user", "parts": [question + 'in brief please']})
        
        response = self.process_question(self.messages)
        self.messages.append({"role": "model", "parts": [response]})

        return response
    
    def gemini_chat(self, question):

        similar_text = ""
        similar_context = self.search_similar_context(self.vector_db, question, 5)
        for context in similar_context:
            similar_text += context.page_content
        
        prompt = f"Use the given below context to answer the user query. {similar_text} Question: {question}?"
        response = self.chat.send_message(prompt + ' in brief please')
        
        return response.text

