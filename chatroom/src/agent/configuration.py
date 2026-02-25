from dataclasses import dataclass

@dataclass
class ChatbotConfig:
    model_name: str
    temperature: float
    system_prompt: str
    assistant_name: str

MODEL_A_CONFIG = ChatbotConfig(
    model_name="gpt-4o-mini",
    temperature=0.7,
    system_prompt="You are a contrarian thinker who callenges assumptions and provides counterintuitive insights.",
    assistant_name="Model A"
)

MODEL_B_CONFIG = ChatbotConfig(
    model_name="gpt-4o-mini",
    temperature=0.9,
    system_prompt="You are an optimistic thinker who sees the glass as half full and provides positive and uplifting insights.",
    assistant_name="Model B"
)