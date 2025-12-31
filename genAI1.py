from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

llm = OllamaLLM(model="phi")

memory = ChatMessageHistory()

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a helpful assistant.

Conversation so far:
{history}

User: {input}
Assistant:
"""
)

chain = prompt | llm

print("Type 'exit' to quit.\n")

while True:
    user_input = input("User: ").strip()
    if user_input.lower() == "exit":
        break

    history_text = "\n".join(
        [f"{m.type.capitalize()}: {m.content}" for m in memory.messages]
    )

    response = chain.invoke({
        "history": history_text,
        "input": user_input
    })

    memory.add_message(HumanMessage(content=user_input))
    memory.add_message(AIMessage(content=response))

    print("\nAI:", response, "\n")
