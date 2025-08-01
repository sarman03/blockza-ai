from langchain_anthropic import ChatAnthropic
from config import ANTHROPIC_API_KEY

def get_llm():
    return ChatAnthropic(
        anthropic_api_key=ANTHROPIC_API_KEY,
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )