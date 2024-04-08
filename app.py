from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

#######
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./testdir/2304.08485.pdf")
pages = loader.load_and_split()

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=pages, embedding=GPT4AllEmbeddings())

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@cl.on_chat_start
async def on_chat_start():
    model = ChatOllama(model="llama2")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a troubleshooting guide for semiconductor manufacturing tools.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = {"docs": format_docs} | prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    question = message.content  
    docs = vectorstore.similarity_search(question)  
    formatted_docs = format_docs(docs) 
    
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content, "context": formatted_docs},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()