"""Interactive personal finance categorizer using RAG over bank statements."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import logging

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROMPTS = {
    "basic": """
You have access to a tool that retrieves financial information from PDFs and XLSX files.
- The bank statements contain transactions and their descriptions.
- Some of the descriptions are not clear and may need to be clarified.
- Answer the question but if the data is missing or ambiguous, say so.

Answer (be specific about what you can/cannot determine):
""",
    "with_categories": """
You have access to a tool that retrieves financial information from PDFs and XLSX files.

When analyzing expenses, categorize them using these categories:
- Groceries
- Restaurants
- Transportation
- Shopping
- Entertainment
- Bills & Utilities
- Healthcare
- Personal Care
- Other

Be consistent with how you categorize similar merchants.
If data is missing or ambiguous, say so explicitly.
""",
    "adaptive": """
You analyze financial transactions from bank statements (PDF and XLSX files).

When categorizing expenses:
1. Identify common expense types from the actual data
2. Create appropriate categories based on spending patterns
3. Categorize each transaction consistently
4. Note any transactions that don't fit clearly

Be specific about uncertainties or missing data.
""",
}

DEFAULT_PROMPT_KEY = "basic"


def get_data_dir() -> Path:
    """Return the data directory path (script-relative)."""
    return Path(__file__).resolve().parent / "data"


def load_excel_documents(folder_path: Path) -> list[Document]:
    """Load Excel files as LangChain Documents using pandas."""
    documents = []
    excel_files = list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.xls"))
    logger.info(f"Found {len(excel_files)} Excel files in {folder_path}")

    for file_path in excel_files:
        try:
            logger.info(f"Loading Excel file: {file_path}")
            df = pd.read_excel(file_path)
            if df.empty:
                logger.warning(f"Empty Excel file: {file_path}")
                continue
            content = df.to_markdown(index=False)
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "row_count": len(df),
                    "columns": df.columns.tolist(),
                },
            )
            documents.append(doc)
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            continue

    logger.info(f"Successfully loaded {len(documents)} Excel files")
    return documents


def load_documents(data_dir: Path) -> list[Document]:
    """Load all PDF and Excel documents from the data directory."""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. Create it and add PDF/XLSX documents."
        )

    pdf_loader = DirectoryLoader(
        path=str(data_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    pdf_docs = pdf_loader.load()
    excel_docs = load_excel_documents(data_dir)
    documents = pdf_docs + excel_docs

    if not documents:
        raise FileNotFoundError(
            f"No documents found in {data_dir}. Add PDF or Excel files to run the analyzer."
        )
    return documents


def build_vector_store(
    documents: list[Document],
    embeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> InMemoryVectorStore:
    """Split documents and build an in-memory vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(documents)
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=splits)
    return vector_store


def choose_prompt() -> str:
    """Show CLI menu and return the selected prompt key (default: basic)."""
    keys = list(PROMPTS.keys())
    print("\nSelect system prompt:")
    for i, key in enumerate(keys, 1):
        default_mark = " (default)" if key == DEFAULT_PROMPT_KEY else ""
        print(f"  [{i}] {key}{default_mark}")
    default_num = keys.index(DEFAULT_PROMPT_KEY) + 1
    while True:
        raw = input(f"Choice [{default_num}]: ").strip()
        if not raw:
            return DEFAULT_PROMPT_KEY
        if raw in keys:
            return raw
        try:
            idx = int(raw)
            if 1 <= idx <= len(keys):
                return keys[idx - 1]
        except ValueError:
            pass
        print("Invalid choice. Enter a number or prompt name.")


def create_categorizer_agent(model, vector_store: InMemoryVectorStore, system_prompt: str):
    """Create the retrieval tool and agent."""

    @tool(response_format="content_and_artifact")
    def retrieve_transactions(query: str):
        """Retrieve transactions from the bank statements."""
        retrieved_docs = vector_store.similarity_search(query, k=4)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    tools = [retrieve_transactions]
    return create_agent(model, tools, system_prompt=system_prompt)


def main() -> None:
    """Run the financial categorizer interactively."""
    data_dir = get_data_dir()
    documents = load_documents(data_dir)

    prompt_key = choose_prompt()
    system_prompt = PROMPTS[prompt_key]

    model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = build_vector_store(documents, embeddings)
    agent = create_categorizer_agent(model, vector_store, system_prompt)

    print("\n" + "=" * 60)
    print("Personal Finance Categorizer")
    print("=" * 60)
    print(f"Loaded {len(documents)} documents from {data_dir}")
    print(f"Prompt: {prompt_key}")
    print("\nExample queries:")
    print("  - Categorize all my expenses")
    print("  - What did I spend on restaurants?")
    print("  - Show me unusual transactions")
    print("\n" + "=" * 60 + "\n")

    while True:
        query = input("\nYour query (or 'quit' to exit): ")

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not query.strip():
            continue

        print("\nAnalyzing...\n")
        for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
