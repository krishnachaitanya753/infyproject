import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Retrieve API key
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key or api_key == "your_nvidia_api_key_here":
    print("Warning: NVIDIA_API_KEY is not set. Please update the .env file.")

client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=api_key
)

print("Streaming completion from Nvidia API (minimaxai/minimax-m2.7)...")
try:
    completion = client.chat.completions.create(
      model="minimaxai/minimax-m2.7",
      messages=[{"role": "user", "content": "Hello! Introduce yourself in one sentence."}],
      temperature=1,
      top_p=0.95,
      max_tokens=8192,
      stream=True
    )

    for chunk in completion:
      if not getattr(chunk, "choices", None):
        continue
      if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
    print() # New line after stream ends

except Exception as e:
    print(f"\nAn error occurred during Nvidia API call: {e}")
    print("Make sure you have set a valid NVIDIA_API_KEY in the .env file.")
