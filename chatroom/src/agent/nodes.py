from agent.configuration import ChatbotConfig
from agent.state import ChatbotState, ChatroomState, ChatMessage, ChatHistory
from langchain_openai import ChatOpenAI

# Factory function to make a respond node
def make_respond_node(config: ChatbotConfig):

    llm = ChatOpenAI(model=config.model_name, temperature=config.temperature)

    def respond(state: ChatbotState):
        # Responds to the query, see if its sender is the user or the assistant, slightly different prompts for each

        query = state["query"]

        # Deserialize if it's a plain dict (LangGraph serializes Pydantic models to dicts)
        if isinstance(query, dict):
            query = ChatMessage(**query)

        if query.sender == "user":
            prompt = f"The user says: {query.content}. Repond to the message"
        else:
            prompt = f"{query.sender} says: {query.content}. Respond to the message, or say nothing if you don't have an answer."

        response = llm.invoke(
            [
                {
                    "role": "system", "content": config.system_prompt
                },
                {
                    "role": "user", "content": prompt
                }
            ]
        )

        return {
            "response": ChatMessage(
                sender=config.assistant_name,
                content=response.content
            )
        }
    
    return respond

def user_input(state: ChatroomState) -> dict:
    msg = state["user_message"]

    # Normalize if coming from Studio as a dict with 'type' instead of 'sender'
    if isinstance(msg, dict):
        msg = ChatMessage(
            sender=msg.get("sender") or msg.get("type", "user"),
            content=msg["content"]
        )

    return {
        "model_a_query": msg,
        "model_b_query": msg,
    }

def prepare_reactions(state: ChatroomState) -> dict:
    return {
        "model_a_query": state["model_b_response"],
        "model_b_query": state["model_a_response"],
    }

# Simple passthrough node to handle human input
def human_turn(state: ChatroomState) -> dict:
    return {}