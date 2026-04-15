"""RAG chain for the Capital Finder Bot with prompt engineering and token optimization."""


from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"

SYSTEM_PROMPT = """You are a concise capital cities expert. Answer the user's question using ONLY the provided context. 
If the answer is not in the context, say "I don't have that information."

Rules:
- Be brief and direct.
- State only the capital name.
- If a country has multiple capitals, mention all of them with their roles.
- Do not repeat the question.

Context:{context}"""


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


def get_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=150,
    )


def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])


def create_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | get_prompt()
        | get_llm()
        | StrOutputParser()
    )
    return chain


def create_chain_with_contexts():
    """Returns a chain that outputs {"answer": str, "contexts": list[str]}.
    This is needed for RAGAS evaluation which requires the retrieved contexts.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def retrieve_and_store(question):
        docs = retriever.invoke(question)
        return {"docs": docs, "question": question}

    def build_answer(inputs):
        docs = inputs["docs"]
        question = inputs["question"]
        context_str = format_docs(docs)
        contexts = [doc.page_content for doc in docs]

        prompt = get_prompt()
        llm = get_llm()
        messages = prompt.invoke({"context": context_str, "question": question})
        response = llm.invoke(messages)

        return {
            "answer": response.content,
            "contexts": contexts,
            "question": question,
        }

    chain = RunnableLambda(retrieve_and_store) | RunnableLambda(build_answer)
    return chain
