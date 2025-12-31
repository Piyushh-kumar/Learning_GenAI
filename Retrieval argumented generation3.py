from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

documents = [Document(page_content=text)]

embeddings = OllamaEmbeddings(model="phi")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(model="phi")

prompt = PromptTemplate(
    input_variables=["context", "questions"],
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

chain =  prompt | llm

question = input("Ask a question: ")

docs =  retriever.invoke(question)

context = "\n\n".join(doc.page_content for doc in docs)

response = chain.invoke({
    "context": context,
    "question": question
})

print("\nAnswer:\n", response)
