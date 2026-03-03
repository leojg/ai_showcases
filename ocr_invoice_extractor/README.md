# Invoice OCR Extractor

This project showcases a LangChain-based tool for extracting structured data from invoice images using OCR. It is designed as a standalone, reusable tool that can be easily integrated into larger LangGraph pipelines.

## Tech Stack

- **LangChain** - LLM chain and tool orchestration
- **OpenAI GPT-4o-mini** - Structured data extraction from raw OCR text
- **pytesseract** - OCR engine for image-to-text conversion
- **Pillow** - Image loading and preprocessing
- **Pydantic** - Structured output validation

## Features

- **Image-based Extraction**: Processes invoice images (JPG, PNG) and extracts structured fields
- **Structured Output**: Returns validated Pydantic models with consistent field naming
- **LangGraph-ready**: Exposed as a `@tool` via a factory function, making it a drop-in node for any LangGraph workflow
- **Flexible Model Injection**: Accepts an external LLM instance, avoiding redundant model instantiation in larger pipelines
- **Graceful Null Handling**: Returns `null` for fields that are absent or ambiguous rather than hallucinating values

## Extracted Fields

| Field | Type | Notes |
|---|---|---|
| `invoice_number` | `str \| null` | Optional |
| `date` | `str` | Required |
| `due_date` | `str \| null` | Optional |
| `business_name` | `str \| null` | Optional |
| `description` | `str \| null` | Inferred from items if not explicit |
| `items` | `list[ItemModel]` | Description and amount per line item |
| `partial_amount` | `float \| null` | Optional subtotal |
| `taxes` | `list[TaxModel] \| null` | Name + percentage or absolute amount |
| `total_amount` | `float` | Required |
| `currency` | `str \| null` | ISO code, only if explicitly stated |

## LangGraph Integration

The tool is exposed via a factory function that injects the shared LLM model:

```python
from invoice_ocr_tool import create_invoice_tool

llm = ChatOpenAI(model="gpt-4o-mini")
invoice_tool = create_invoice_tool(llm)

# Use directly in a LangGraph node
def invoice_node(state):
    return invoice_tool.invoke(state["image_path"])
```

This pattern avoids re-instantiating the model on each call and keeps the tool signature clean (`image_path: str`) and serialization-safe.

## Technical Decisions

### Why LangChain and not LangGraph?
This is a **linear pipeline** with no branching or persistent state: load image → OCR → LLM extraction → structured output. LangGraph adds unnecessary complexity for straight chains. LangGraph is reserved for the larger financial reconciliation pipeline where this tool is consumed as a node.

### Why use an LLM at all?
For clean, machine-generated invoices, regex-based parsing could work. The LLM adds value when:
- OCR output is messy or has layout artifacts
- Invoice formats vary across vendors
- Field semantics are ambiguous (e.g. "net 30", "payment by", "due date" all mean the same thing)

### Currency Handling
Currency is intentionally left as `null` when not explicitly stated in the invoice. The `$` symbol alone is not a reliable indicator, it is used by Argentina, Uruguay, Colombia, USA, and others. Resolving currency ambiguity is a **graph-level concern**, where broader document context (account region, other invoices in the batch, user settings) can make a reliable determination.

> **Lesson learned:** Even with explicit null instructions in the system prompt, `gpt-4o-mini` will infer currency from contextual clues like business names or locale hints in the text. Prompt phrasing matters, vague instructions like "leave it as null if ambiguous" are insufficient. A more forceful explicit instruction was required:
> ```
> IMPORTANT: currency MUST be null unless an explicit currency code appears 
> in the invoice (USD, EUR, ARS, etc.). The $ symbol alone is NOT sufficient.
> ```
> Additionally, Pydantic field `description` examples (e.g. `"e.g. USD, EUR, UYU"`) can inadvertently bias the model toward guessing. Keep field descriptions neutral.

## Usage

```bash
# Install dependencies
pip install langchain langchain-openai pytesseract pillow pydantic python-dotenv

# Install system dependencies
sudo apt install tesseract-ocr

# Set up API keys
cp .env.example .env
# Edit .env: add OPENAI_API_KEY

# Add invoice images
mkdir -p data
cp /path/to/invoice.jpg ./data/

# Run the extractor
python invoice_ocr_tool.py
```

## Requirements

- Python 3.10+
- OpenAI API key

## Future Improvements

- [ ] Preprocessing pipeline for skewed or low-resolution scans (contrast, deskew)
- [ ] OCR confidence scoring to flag low-quality extractions
- [ ] Support for multiple OCR backends (Google Vision, AWS Textract) with a common interface
- [ ] Batch processing of multiple images

## License

Apache