from langchain_anthropic import ChatAnthropic
from config import ANTHROPIC_API_KEY

def get_llm():
    return ChatAnthropic(
        api_key=ANTHROPIC_API_KEY,
        model="claude-3-opus-20240229",
        temperature=0.1,
        max_tokens=2000
    )