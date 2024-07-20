from langchain_community.vectorstores.pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# I used the abdulrahman code and just instead of deleting and creating the vector db everytime we run the app (session)
# just retrieve the created vector db from pinecone
# this is the key that will be used to access the gp-iti index (pinecone vector database)
# call load_vector_db and assign its return to variable (vector = load_vector_db())
# than you can pass vector variable to ask_and_get_answer(vector, q) function.

class GPT:
    def __init__(self):
        # can add pinecone_key to .env file and remove it from here, and also openai_key
        os.environ['PINECONE_API_KEY'] = 'c8519fb3-b5b7-461e-afa7-0090e4a5fa43'
        load_dotenv(find_dotenv(), override=True)
        self.vector_db = self.load_vector_db()
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.client = OpenAI()

    def load_vector_db(self):
        return Pinecone.from_existing_index("gp-iti", OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536))

    def ask_and_get_answer(self, q, k=3):
        results = self.vector_db.similarity_search_with_score(q)
        self.messages.append({"role": "user", "query": q})

        if not results or results[0][1] < 0.4:
            llm = ChatOpenAI(model='gpt-4', temperature=1)
            return llm.invoke(q).content
        else:
            llm = ChatOpenAI(model='gpt-4', temperature=0.6)
            retriever = self.vector_db.as_retriever(search_type='similarity', search_kwargs={'k': k})
            chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            answer = chain.invoke(q)
            return answer['result']
        
# Testing
#    
# from Model.GPT import *
# gpt = GPT()
# answer = gpt.ask_and_get_answer(quest)

     
    def ask_and_get_answer_remember_history(self, q, k=3):
        results = self.vector_db.similarity_search_with_score(q)

        if not results or results[0][1] < 0.4:
            # llm = ChatOpenAI(model='gpt-4', temperature=1)
            # return llm.invoke(q).content

            self.messages.append({"role": "user", "content": q})
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages= self.messages
            )
    
        else:
            llm = ChatOpenAI(model='gpt-4o', temperature=0.6)
            retriever = self.vector_db.as_retriever(search_type='similarity', search_kwargs={'k': k})
            chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            result = chain.run(q)

            self.messages.append({"role": "system", "content": result})
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages= self.messages
            )
            
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})

        return response.choices[0].message.content

    def generate(self, question, lang = 'en'):

        return self.ask_and_get_answer_remember_history(question)
