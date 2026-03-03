from pydantic import BaseModel, Field
from langchain_core.tools import tool
from PIL import Image
import pytesseract
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

from dotenv import load_dotenv

load_dotenv()

class ItemModel(BaseModel):
    description: str
    amount: float

class TaxModel(BaseModel):
    name: str
    percentage: float | None = Field(None, description="e.g. 21.0 for 21%")
    amount: float | None = Field(None, description="absolute tax amount if percentage not shown")

class InvoiceData(BaseModel):
    invoice_number: str | None = None
    date: str
    due_date: str | None = None
    business_name: str | None = None
    description: str | None = Field(None, description="inferred from items if not explicit")
    items: list[ItemModel]
    partial_amount: float | None = None
    taxes: list[TaxModel] | None = None
    total_amount: float
    currency: str | None = Field(None, description="ISO currency code, only if explicitly stated in the invoice")

def extract_text_from_image(image_path: str) -> str:
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def extract_invoice_data_from_text(raw_text: str, model: BaseChatModel) -> InvoiceData:
    """Extracts structured invoice data from text."""

    structured_llm = model.with_structured_output(InvoiceData)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        """
            You are an invoice data extraction assistant. 
            Extract structured data from the provided invoice text. 
            If a field is not present or ambiguous, leave it as null. 
            IMPORTANT: currency MUST be null unless an explicit currency code 
            appears in the invoice (USD, EUR, ARS, UYU, etc.). 
            The $ symbol alone is NOT sufficient; leave currency as null.

        """),
        ("human", "{raw_text}"),
    ])

    chain = prompt | structured_llm
    return chain.invoke({"raw_text": raw_text})


def create_invoice_tool(model: BaseChatModel):
    @tool
    def extract_invoice_data(image_path: str) -> str:
        """Extracts structured invoice data from an image."""
        raw_text = extract_text_from_image(image_path)
        invoice_data = extract_invoice_data_from_text(raw_text, model)
        return invoice_data.model_dump()

    return extract_invoice_data

def main() -> None:
    """Run the invoice data extractor"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    invoice_tool = create_invoice_tool(llm)

    print(invoice_tool.invoke("data/ticket.png"))

if __name__ == "__main__":
    main()
