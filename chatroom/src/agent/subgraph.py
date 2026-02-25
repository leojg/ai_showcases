from langgraph.graph import StateGraph, START, END
from agent.state import ChatbotState
from agent.configuration import ChatbotConfig, MODEL_A_CONFIG, MODEL_B_CONFIG
from agent.nodes import make_respond_node

def make_chatbot_subgraph(config: ChatbotConfig):

    respond = make_respond_node(config)

    graph = StateGraph(ChatbotState)

    graph.add_node("respond", respond)

    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)

    return graph.compile()

model_a_subgraph = make_chatbot_subgraph(MODEL_A_CONFIG)
model_b_subgraph = make_chatbot_subgraph(MODEL_B_CONFIG)