from dotenv import load_dotenv


import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

print("OpenAI API Key:" , openai_api_key)
print("Anthropic API Key:" , anthropic_api_key)