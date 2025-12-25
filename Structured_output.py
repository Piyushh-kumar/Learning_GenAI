from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import json

llm = OllamaLLM(model="phi")

prompt = PromptTemplate(
    input_variables=["cuisine"],
    template="""
You are a strict JSON generator.

Rules:
1. Output ONLY valid JSON.
2. Do NOT add extra fields.
3. Do NOT add explanations.
4. Follow the schema EXACTLY.

Return JSON in this exact format:
{{
  "restaurant_name": "...",
  "tagline": "..."
}}

Cuisine: {cuisine}
"""
)

chain = prompt | llm

cuisine = input("Enter cuisine: ").strip()

MAX_RETRIES = 3
attempt = 0

while attempt < MAX_RETRIES:
    response = chain.invoke({"cuisine": cuisine})
    print(f"\nAttempt {attempt + 1} raw output:\n{response}")

    try:
        parsed = json.loads(response)
        print("\nParsed JSON output:\n", parsed)
        break
    except json.JSONDecodeError:
        print("Invalid JSON. Retrying...\n")
        attempt += 1

if attempt == MAX_RETRIES:
    print("Failed to get valid JSON after retries.")

