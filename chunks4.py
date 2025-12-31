# Import the text splitter used for chunking text
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------- RAW TEXT (SIMULATING A FILE) --------------------

# This is a long text, similar to what you would read from a file
text = """
LangChain is a framework for building applications using large language models.
It provides tools for chaining prompts, managing memory, and integrating external data.
Retrieval Augmented Generation (RAG) helps models answer questions using external knowledge.
Chunking is an important step to ensure better retrieval accuracy.
"""

# -------------------- CREATE TEXT SPLITTER --------------------

# chunk_size = maximum number of characters in one chunk
# chunk_overlap = number of characters shared between adjacent chunks
# RecursiveCharacterTextSplitter tries to split intelligently:
# first by paragraphs, then sentences, then words if needed
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=30
)

# -------------------- SPLIT TEXT INTO CHUNKS --------------------

# split_text() breaks the long text into smaller overlapping chunks
chunks = text_splitter.split_text(text)

# -------------------- OUTPUT CHUNKS --------------------

# Print each chunk so we can see how chunking actually works
for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
