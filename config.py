import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DIRECTORY_API_URL = "https://api.blockza.io/api/directory"
