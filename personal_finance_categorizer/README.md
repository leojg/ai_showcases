# Agentic Personal Finance RAG Demo project

This project aims to showcase the implementation of an Agentic RAG system using LangChain. It focuses on extracting information from various sources of financial and bank statements and answer queries about spending habits, spending categories and anomalous expenses.

## Tech Stack

- **LangChain** - Agent orchestration and RAG framework
- **Claude Sonnet 4.5** - Language model for reasoning and generation
- **OpenAI Embeddings** - Semantic search over documents
- **Pandas** - Excel file processing
- **PyPDF** - PDF document loading

## Features

- **Multi-Source Integration**: Seamlessly combines PDF and Excel bank statements from different sources
- **Multi-Currency Support**: Analyzes accounts in different currencies (USD, UYU) with automatic conversion
- **Agentic RAG**: AI agent decides when and how to retrieve information from documents
- **Smart Categorization**: Automatically creates logical expense categories based on actual spending patterns
- **Anomaly Detection**: Identifies unusual transactions, large expenses, and potential issues
- **Cross-Document Reasoning**: Synthesizes information across multiple statements and formats
- **Actionable Insights**: Provides recommendations and next steps, not just data analysis
- **Uncertainty Handling**: Explicitly acknowledges incomplete or ambiguous data rather than hallucinating
- **Merchant Intelligence**: Handles unclear merchant names and identifies patterns across variations
- **Spending Analysis**: 
  - Total spending by category
  - Top merchants by amount
  - Pattern recognition (recurring payments, transfers)
  - Spending trends and observations

## What Makes This Different

Unlike simple expense trackers, this system:

- **Handles real messy data**: Works with actual bank statements that have unclear merchant names, mixed currencies, and varying formats
- **Thinks like a financial analyst**: Not just categorizing, but detecting anomalies and providing recommendations
- **Transparent about limitations**: Acknowledges when data is incomplete rather than making assumptions
- **Production-ready approach**: Uses pandas for Excel processing (lightweight, fast) vs heavy document parsing frameworks
- **End-to-end insights**: From raw PDFs to actionable financial analysis in one query

## Technical Decisions

### Prompt Engineering
Three prompt strategies are available:
- **basic**: Flexible, uncertainty-aware (default)
- **with_categories**: Explicit taxonomy for consistency
- **adaptive**: Agent-generated categories from data

The production default is "basic" for flexibility, but you can experiment 
with others via the interactive prompt selection.

### Excel Processing
Uses **pandas** for Excel file loading rather than `UnstructuredExcelLoader`.

**Rationale:**
- Lightweight (pandas is standard; UnstructuredExcelLoader requires 100+ MB of dependencies)
- Clean output (`df.to_markdown()` produces well-formatted tables for LLMs)
- Better control for validation and error handling
- Faster for structured tabular data

**Trade-off:** Works best for standard Excel tables. Complex spreadsheets with merged cells or unusual formatting may benefit from UnstructuredExcelLoader.

## Example Output:

**Query**
``` 
"What did I spend on groceries?"
``` 

**Response**
```
Based on the bank statement transactions, I can identify several grocery-related purchases. Here's what you spent on groceries:

**Confirmed Grocery Purchases:**
- **MADRE TIERRA**: $7.36 (Jan 9)
- **SUPER PANDIT**: $35.23 + $22.55 = $57.78 (Jan 5)
- **PUESTO PASTO**: $10.20 (Jan 5)
- **MERCADITO LO**: (amount not fully shown but appears to be a small grocery store)
- **EL REVOLTIJO**: $16.34 (Jan 15) + $757.00 (Jan 16) = $773.34
- **MERPAGO*CADO** (likely Mercado/grocery): $17.38 (Jan 12)

**Likely Grocery-Related:**
- **PASTAS PALER** (pasta shop): (purchase amount not shown in results)
- **EL FRUTO**: $124.00 (Jan 2) - could be a fruit/produce store

**Total for confirmed groceries: approximately $866.06** (excluding unclear entries)

**Note:** Some transaction amounts are not fully visible in the statement, and some merchant names are abbreviated or unclear (like "MERPAGO*CADO"). The actual total could be higher if additional entries like PASTAS PALER, MERCADITO LO, and others are included. The currency appears to be in USD or UYU based on the statement format.
```

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp ../.env.example .env
# Edit .env: add OPENAI_API_KEY and ANTHROPIC_API_KEY

# Add your bank statements
mkdir -p data
cp /path/to/statements/*.pdf ./data/
cp /path/to/statements/*.xlsx ./data/

# Run the categorizer
python categorizer.py
```

The script will:
1. Let you choose a prompt strategy (basic, with_categories, adaptive)
2. Load and process your documents
3. Enter interactive mode for queries

## Requirements

- Python 3.10+
- OpenAI API key (for embeddings)
- Anthropic API key (for Claude)

## Future Improvements

- [ ] Export categorized data to CSV
- [ ] Persistent vector storage (Chroma)
- [ ] Budget tracking and alerts
- [ ] Multi-month trend analysis
- [ ] Receipt matching to statement transactions

## License

Apache