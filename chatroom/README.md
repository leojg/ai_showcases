# Chatroom Sample

This project aims to showcase the implementation of an chatbot debate system using LangGraph.

It consists of two separate chatbots with distinct personalities that interact with each other and a human participant. Both models first reply to the human query and then react to the chatbot response to that query, simulating a three-way conversation.

## Tech Stack

- **LangGraph**: Agent orchestration, graph definition and subgraph management.
- **LangGraph Studio**: Local development and visualization
- **OpenAI 4o-mini**: Language model powering both chatbot subgraphs.
- **LangChain OpenAI**: LLM client integration
- **Pydantic**: Domain entities definition

## Features
- **Debate system**: Provides a threeway debate system between human and two models
- **Parallel Execution**: Both models respond and react simultaneously using a fan-out pattern
- **Multi turn conversation**: Human can continue the conversation across multiple turns
- **Subgraph Architecture**: Each chatbot is an independent LangGraph subgraph with its own configuration
- **Human-in-the-loop**: Graph pauses after each round and waits for the human input before continuing
- **Configurable Personas**: Each model has its own system prompt, temperature and model settings.

## What Makes This Different

Unlike a standard chatbot:

- **Models talk to each other**: Each model receives the other's response and reacts to it, not just the human message.
- **Parallel processing**: Both models run concurrently in each pass, not sequentially
- **Clean subgraph separation**: Each chatbot is fully encapsulated with its own state and configuration, making it easy to swap models or prompts independently

# Graph Architecture

![Chatroom Graph Architecture](./static/chatroom_graph.png)


# Technical Decisions

### Subgraph per model

Each chatbot is a compiled LangGraph subgraph with a single respond node. The same node handles both the initial response and the reaction pass, the difference is what's passed as `query`(the user message vs the other model response) 

This keeps each chatbot self-contained and independently configurable. Adding a third model is as simple as adding a new subgraph and wiring it into the main graph.

### State Design

The main graph `ChatroomState` own the conversation history and coordinates the two models. Subgraphs `ChatbotState` only see their current query and chat history, no cross-model state leakage

### Factory Pattern

Both subgraphs nodes and subgraphs themselves are created via factory functions(`make_respond_node`, `make_chatbot_subgraph`), keeping the logic DRY while allowing per-model configuration.

### Example Output

User
```
  What is the meaning of life?
```
Model A (Contrarian):
```
The question itself is flawed — assuming life has an inherent "meaning" 
presupposes a designer with intent. Perhaps the more honest answer is 
that meaning is something we impose, not discover.
```
Model B (Optimist):
```
What a beautiful question! I believe the meaning of life is connection — 
to others, to ideas, to experiences that make us feel alive. Every day 
is an opportunity to create your own meaning!
```
Model A reacts to Model B:
```
"Connection" is a comforting narrative, but it sidesteps the harder 
question. Connection to what end? Feelings of aliveness don't constitute 
meaning — they're just neurochemistry.
```
Model B reacts to Model A:
```
I love the depth you bring! While I respect the philosophical rigor, 
I think dismissing neurochemistry undersells it — those feelings of 
connection are the very fabric of human experience!
```

### Usage

```
# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp ../.env.example .env
# Edit .env: add OPENAI_API_KEY

# Run with LangGraph Studio
langgraph dev
```
Then open LangGraph Studio, create a new thread, and submit your first message via the Input panel.

### Future Improvements

- [ ] Conversation history passed to models for multi-turn memory
- [ ] Private subgraph state with per-model conversation summary, used to personalize each model's interpretation of the conversation history before responding
- [ ] Empty response handling (model opts out if nothing to add)
- [ ] Custom UI (Streamlit or Chainlit) with side-by-side model display
- [ ] Support for additional LLM providers (Anthropic, Gemini)
- [ ] Configurable number of reaction rounds
- [ ] Export conversation transcript
- [ ] Add longterm memory to support 
- [ ] Persistent conversation storage using LangGraph checkpointer (SQLite local, Postgres for production)