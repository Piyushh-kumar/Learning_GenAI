rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a query rewriting assistant. Given the user's query, rewrite it to be more specific and clear.
User Query: {question}

Rewritten Query:"""
)

rewriter_chain = rewrite_prompt | llm

user_question = input("Ask a question: ")

# Rewrite question for retrieval
rewritten_query = rewriter_chain.invoke({
    "question": user_question
})

print("\nRewritten query:\n", rewritten_query)





#Simple multi-step example (conceptual code)

sub_questions = [
    "What is Retrieval Augmented Generation?",
    "What is fine-tuning in LLMs?",
    "Differences between RAG and fine-tuning"
]

all_docs = []

for q in sub_questions:
    docs = retriever.invoke(q)
    all_docs.extend(docs)

context = "\n\n".join(doc.page_content for doc in all_docs)
