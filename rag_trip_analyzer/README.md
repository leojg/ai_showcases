# Agentic RAG Demo project

This project aims to showcase the implementation of an Agentic RAG system using LangChain. It focuses on extracting information about a previous trip and answering queries about dates, location, cost, and people participating in the trip.

## Features

- **Agentic RAG**: Uses an AI agent that decides when and how to retrieve information
- **Multi document reasoning**: Synthesizes information across different document types
- **Uncertainty handling**: Explicitly states when data is incomplete or ambiguous
- **Source attribution**: Shows which documents were used to answer the queries

## Example Output:

**Query**
``` 
What was the duration of the trip?
Where did I go on my trip?
How many people were on the trip?
What was the cost of the whole trip?
``` 


**Response**
``` 
Based on the travel documents I found, here's what I can determine about your trip:

## **Duration of the Trip:**
**3 nights / 4 days** - From January 9 to January 12, 2026
- Departure: January 9, 2026 (left Montevideo at 16:45)
- Return: January 12, 2026 (returned to Montevideo)

## **Where You Went:**
**Buenos Aires, Argentina** (from Montevideo, Uruguay)
- You traveled via Colonia Express ferry from Montevideo → Colonia → Buenos Aires
- Stayed at ###### Hotel Buenos Aires (located at ###### ###### ######)
- You also had a bar tour activity on January 10, 2026

## **Number of People:**
**4 people total:**
- ###### ######
- ###### ######
- 2 additional adults (names confirmed on hotel booking for 4 adults total)

The ferry tickets show 2 passengers (###### ###### and ###### ######), and the hotel booking was for "4 adultos" in 2 rooms.

## **Total Cost of the Trip:**
I can identify these specific costs:
- **Ferry transportation (Colonia Express):** 7,198.00 UYU (approximately ~$118 USD at exchange rates shown)
- **Hotel accommodation:** US$405.08 (subtotal for 2 rooms, 3 nights) + US$18 municipal tax = **~US$423.08 total**
- **Bar tour activity:** 8,120.00 $U (Uruguayan pesos)

**However, I cannot determine the complete total trip cost** because:
- The documents show various expenses on bank statements but don't clearly separate trip costs from regular daily expenses
- There may be additional meals, transportation within Buenos Aires, and other activities not fully detailed
- The currency conversions between UYU and USD vary throughout the documents

The major documented expenses total approximately **US$540-600**, but this is incomplete data.
```

_Some names and locations were redacted to keep privacy and anonymity._


## Tech Stack:

- **LangChain**: Document processing and agent orchestration
- **Claude (Anthropic)**: Language model for reasoning and generation
- **OpenAI Embeddings**: Semantic search over documents
- **PyPDF**: PDF parsing

## How it works

1. **Document Loading**: Reads all PDFs from the `data/` folder
2. **Chunking**: Splits the documents into searchable chunks (1000 chars with 200 overlap)
3. **Embedding**: Creates vector embeddings using OpenAI model
4. **Agent Creation**: Set up a Claude powered agent with retrieval tool
5. **Query Processing**: Agent decides when to search, synthesizes results, handles ambiguity

## Setup Instructions

### Prerequisites
- Python 3.10+
- OpenAI API key
- Anthropic API key

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/leojg/ai_showcases.git
cd ai_showcases/rag_trip_analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up API keys:**
```bash
cp ../.env.example .env
# Edit .env and add your keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

4. **Add your documents:**
```bash
# Place your PDF documents in the data/ folder
cp /path/to/your/pdfs/*.pdf ./data/
```

5. **Run:**
```bash
python trip_rag_analyzer.py
```

## Cost Estimate

Approximate cost per run with ~10 PDF pages:
- Embeddings: ~$0.01 (one-time per session)
- Claude queries: ~$0.05 per complex query

## Project Learnings

- **Data Quality**: Travel documents often are incomplete or scattered
- **Data Variability**: Data such as dates can be found in multiple formats across different documents
- **Reconciliation**: Matching booked amounts with actual charges
- **Ambiguity Handling**: Important to acknowledge data gaps rather than hallucinate

**Key insights**:

- Agentic RAG is a more flexible approach compared to basic retrieval chains
- Proper prompt engineering significantly improves the output quality
- Multi document reasoning requires carefully chunk sizing and retrieval tuning 

## Future Improvements

- [ ] Add Chroma for persistent vector storage
- [ ] Extract structured data (dates, amounts, merchants)
- [ ] Add expense categorization
- [ ] Compare booked vs actual costs (reconciliation)
- [ ] Support image-based PDFs (OCR)
- [ ] Add web interface (Streamlit/Gradio)

**Note**: This is a learning project demonstrating RAG concepts. Contributions and feedback welcome!