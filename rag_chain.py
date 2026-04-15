"""RAG chain for the Capital Finder Bot with prompt engineering and token optimization."""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"

SYSTEM_PROMPT = """You are a concise capital cities expert. Answer the user's question using ONLY the provided context. If the answer is not in the context, say "I don't have that information."

Rules:
- Be brief and direct.
- State the capital name first, then add one key fact if relevant.
- Do not repeat the question.

Context:
{context}"""

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


def create_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=150,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
