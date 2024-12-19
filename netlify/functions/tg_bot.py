from flask import Flask, request
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import  ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage


api_key = os.environ['GROQ_API_KEY']

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = api_key
)

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever()

system_prompt = (
    """ Act like you are a chatbot that is deployed to enlighten people about their rights and laws in Nigeria.
        Use the following pieces of context to answer the question at the end.
        The documents provided include the Nigerian constitution, Criminal Code Act, Cybercrime Act, Copyright act and Central Bank of Nigeria Act 
        when the user mentions money, they are refering naira.
        When answering, don't say things like "based on the provided document", just mention or reference the specific document
        Don't answer the question if it's not within the ambits of legal rights or law.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Do not respond to abusive language.
        Explain briefly except the user says otherwise.
        Give scenarios where necessary.
        {context}
        Question: {input}
        Helpful Answer:
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)



qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []


app = Flask(__name__)

# Initialize Telegram bot
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']

async def start(update, context):
    await update.message.reply_text("Hello! I'm your legal buddy.")

async def handle_message(update: Update, context) -> None:
    user_message = update.message.text
    # print(user_message)
    reply = rag_chain.invoke({"input": user_message, "chat_history": chat_history})
    response = reply["answer"]
    chat_history.extend(
        [
            HumanMessage(content=user_message),
            AIMessage(content=response)
        ]
    )
    
    await update.message.reply_text(response)



# Define a simple command handler
async def start(update: Update, context):
    await update.message.reply_text("Hello! I am your legal buddy.")

# Main function to run the bot
def main():
    # Replace 'YOUR_BOT_TOKEN' with your actual token
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers to the application
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()



