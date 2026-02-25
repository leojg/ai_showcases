from pydantic import BaseModel
from typing import TypedDict, Annotated

class ChatMessage(BaseModel):
    sender: str # "user" or the assistant name
    content: str

class ChatHistory(BaseModel):
    messages: list[ChatMessage] = []

    def add(self, message: ChatMessage) -> "ChatHistory":
        return ChatHistory(messages=[*self.messages, message])

# Reducer function to merge two chat histories
def merge_history(old: ChatHistory, new: ChatHistory) -> ChatHistory:
    return ChatHistory(messages=[*old.messages, *new.messages])

class ChatroomState(TypedDict):
    user_message: ChatMessage
    model_a_query: ChatMessage
    model_b_query: ChatMessage
    model_a_response: ChatMessage
    model_b_response: ChatMessage
    conversation_history: Annotated[ChatHistory, merge_history]

class ChatbotState(TypedDict):
    query: ChatMessage
    response: ChatMessage
    chat_history: ChatHistory