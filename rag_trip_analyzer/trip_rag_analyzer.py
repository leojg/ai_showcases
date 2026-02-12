from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent

model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore(embedding=embeddings)

# Data path relative to this script so it works from any CWD
_data_dir = Path(__file__).resolve().parent / "data"
loader = DirectoryLoader(path=str(_data_dir), glob="**/*.pdf", loader_cls=PyPDFLoader)

if not _data_dir.is_dir():
    raise FileNotFoundError(
        f"Data directory not found: {_data_dir}. Create it and add PDF documents."
    )
documents = loader.load()
if not documents:
    raise FileNotFoundError(
        f"No PDFs found in {_data_dir}. Add .pdf files to run the analyzer."
    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(documents)
print(f"Split documents into {len(all_splits)} sub-documents.")

documents_ids = vector_store.add_documents(documents=all_splits)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]

prompt = (
    f"""
    You have access to a tool that retrieves travel information from PDFs.
    - The pdfs may be incomplete
    - Answer the question but if the data is missing or ambiguous, say so.

    Answer (be specific about what you can/cannot determine):
    """
)

agent = create_agent(model, tools, system_prompt=prompt)

query = (
    "What was the duration of the trip?\n\n"
    "Where did I went on my trip?\n\n"
    "How many people were on the trip?\n\n"
    "What was the cost of the whole trip?"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
): event["messages"][-1].pretty_print()