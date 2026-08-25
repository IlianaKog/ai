import os
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Setup API Key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# --- OFFLINE PHASE: Ingestion & Indexing ---

# Simulated enterprise documents / knowledge base
raw_documents = [
"""GenAI Practice delivers enterprise-scale AI solutions.
Our focus areas include intelligent document processing, agentic SDLC automation,
and secure enterprise knowledge retrieval using RAG architectures.""",
"""Security and compliance are mandatory for all client deployments.
All solutions must adhere to strict zero-retention policies, data isolation,
and automated hallucination guardrails before production release."""
]

# Step 1: Chunking
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=200,
chunk_overlap=30
)
chunks = text_splitter.create_documents(raw_documents)

# Step 2: Embedding Generation & Vector Store Indexing
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
documents=chunks,
embedding=embedding_model
)

# Set up Retriever (top-k search)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- ONLINE PHASE: Retrieval & Generation ---

# Step 3: Prompt Augmentation Template
system_prompt = """You are a helpful assistant. Use ONLY the following pieces of retrieved context to answer the question.
If you do not know the answer based strictly on the context, say that you do not know. Do not hallucinate.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
("system", system_prompt),
("human", "{question}")
])

# Step 4: LLM Generation setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def format_docs(docs):
return "\n\n".join(doc.page_content for doc in docs)

# Step 5: LCEL (LangChain Expression Language) Pipeline
rag_chain = (
{"context": retriever | format_docs, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)

# --- Execution ---
if __name__ == "__main__":
query = "What are the security requirements for client deployments?"

print(f"Query: {query}\n")
response = rag_chain.invoke(query)
print(f"Answer:\n{response}")
