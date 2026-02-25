"""
    Makes a chatroom graph for the chatroom agent
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when this file is loaded by path (e.g. langgraph dev)
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from langgraph.graph import StateGraph, START, END
from agent.state import ChatroomState
from agent.subgraph import model_a_subgraph, model_b_subgraph
from agent.nodes import user_input, prepare_reactions, human_turn

# Mapper function to map the graph and subgraph states keys
def make_subgraph_node(subgraph, query_key: str, response_key: str):

    def node(state: ChatroomState) -> dict:
        result = subgraph.invoke(
            {
                "query": state[query_key],
                "chat_history": state["conversation_history"],
            }
        )
        return {
            response_key: result["response"],
        }
    
    return node

model_a_node = make_subgraph_node(model_a_subgraph, "model_a_query", "model_a_response")
model_b_node = make_subgraph_node(model_b_subgraph, "model_b_query", "model_b_response")

def make_chatroom_graph():
    graph = StateGraph(ChatroomState)

    graph.add_node("user_input", user_input)
    graph.add_node("model_a_respond", model_a_node)
    graph.add_node("model_a_react", model_a_node)

    graph.add_node("model_b_respond", model_b_node)
    graph.add_node("model_b_react", model_b_node)

    graph.add_node("prepare_reactions", prepare_reactions)
    graph.add_node("human_turn", human_turn)

    graph.add_edge(START, "user_input")

    graph.add_edge("user_input", "model_a_respond")
    graph.add_edge("user_input", "model_b_respond")

    graph.add_edge("model_a_respond", "prepare_reactions")
    graph.add_edge("model_b_respond", "prepare_reactions")

    graph.add_edge("prepare_reactions", "model_a_react")
    graph.add_edge("prepare_reactions", "model_b_react")

    graph.add_edge("model_a_react", "human_turn")
    graph.add_edge("model_b_react", "human_turn")

    graph.add_edge("human_turn", END)

    # Interrupt before the human turn to allow for human input.
    return graph.compile(interrupt_before=["human_turn"])

graph = make_chatroom_graph()