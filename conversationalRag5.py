from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

llm = OllamaLLM(model="phi")

# Embedding model used for retrieval
embeddings = OllamaEmbeddings(model="phi")

chat_history = [
    "What is Retrieval Augmented Generation?",
    "How is it different from fine-tuning?"
]
user_question = "Why does it help reduce hallucinations?"

condense_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Given the conversation below and a follow-up question,
rewrite the follow-up question so it is a standalone question.

Conversation:
{chat_history}

Follow-up question:
{question}

Standalone rewritten question:
"""
)

# Create a chain to rewrite the question
condense_chain = condense_prompt | llm

history_text = "\n".join(chat_history)

# Rewrite the user's question using chat history
standalone_question = condense_chain.invoke({
    "chat_history": history_text,
    "question": user_question
})

print("\nRewritten retrieval query:\n", standalone_question)
text = """
Retrieval Augmented Generation reduces hallucinations by grounding responses
in external documents instead of relying only on model memory.
"""

# Wrap text into a Document
documents = [Document(page_content=text)]

# Create vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

docs = retriever.invoke(standalone_question)

# Combine retrieved docs into context
context = "\n\n".join(doc.page_content for doc in docs)

answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)

answer_chain = answer_prompt | llm

final_answer = answer_chain.invoke({
    "context": context,
    "question": user_question
})

print("\nFinal Answer:\n", final_answer)