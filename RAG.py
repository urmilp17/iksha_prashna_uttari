import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_astradb import AstraDBVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
GOOGLE_API_KEY = "AIzaSyD2-J-odcMaAX5kV_L73Om4RsnX6o5kVNo"
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:bWWcidaHklrOhifcvpdpHrQz:3b90d8e25c75dec85ede15d7004570eb920c23a44b7499f1b359f520ad89825a"
ASTRA_DB_ID = "c93eb10d-bdea-4d33-9ad7-9a670676591f"
ASTRA_DB_API_ENDPOINT = "https://c93eb10d-bdea-4d33-9ad7-9a670676591f-us-east1.apps.astra.datastax.com"
ASTRA_DB_NAMESPACE = "scriptures"
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = AstraDBVectorStore(
    collection_name="astra_vector_langchain",
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context." 
    Do not provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001", temperature=0.3)

    chain = create_stuff_documents_chain(llm, prompt)

    return chain


def user_input(user_question):
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain.invoke(
        {"context": docs, "question": user_question}, return_only_outputs=True)

    return response
