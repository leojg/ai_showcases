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
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_excel_documents(folder_path: str):
    """
        Load Excel files as LangChain Documents using pandas.
        
        Args:
            data_dir: Directory containing Excel files
            
        Returns:
            List of Document objects
    """
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

            # Convert to markdown so it's cleaner for the LLM
            content = df.to_markdown(index=False)

            # 
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "row_count": len(df),
                    "columns": df.columns.tolist(),
                }
            )

            documents.append(doc)

        except Exception as e:
            logger.error(f"✗ Failed to load {file_path.name}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(documents)} Excel files")
    return documents




model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore(embedding=embeddings)

# Data path relative to this script so it works from any CWD
_data_dir = Path(__file__).resolve().parent / "data"

if not _data_dir.is_dir():
    raise FileNotFoundError(
        f"Data directory not found: {_data_dir}. Create it and add PDF and XLSX documents."
    )

# Load PDFs
pdf_loader = DirectoryLoader(
    path=str(_data_dir),
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
pdf_docs = pdf_loader.load()

# Load XLSX
excel_docs = load_excel_documents(_data_dir)

# Combine
documents = pdf_docs + excel_docs

if not documents:
    raise FileNotFoundError(
        f"No documents found in {_data_dir}. Add files to run the analyzer."
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
def retrieve_transactions(query: str):
    """Retrieve transactions from the bank statements."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_transactions]



prompts = {
    "basic": 
    """
        You have access to a tool that retrieves financial information from PDFs and XLSX files.
        - The bank statements contain transactions and their descriptions.
        - Some of the descriptions are not clear and may need to be clarified.
        - Answer the question but if the data is missing or ambiguous, say so.

        Answer (be specific about what you can/cannot determine):    
    """,
    "with_categories":
    """
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
    "adaptive":
    """
        You analyze financial transactions from bank statements (PDF and XLSX files).
    
        When categorizing expenses:
        1. Identify common expense types from the actual data
        2. Create appropriate categories based on spending patterns
        3. Categorize each transaction consistently
        4. Note any transactions that don't fit clearly
        
        Be specific about uncertainties or missing data.
    """
}

query = """
Categorize all my expenses from the bank statements.
Show me:
    - Total spending by category
    - Top 5 merchants
    - Any unusual expenses
"""

for name, prompt in prompts.items():
    print("\n" + "="*60)
    print(f"TESTING: {name}")
    print("="*60)

    agent = create_agent(model, tools, system_prompt=prompt)

    print(f"\nQuery: {query}\n")
    print("Response:")

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ): event["messages"][-1].pretty_print()
    
    print("\n" + "="*60)
    input("Press Enter to continue to next prompt...")
