import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key or api_key == "your_openrouter_api_key_here":
    print("Warning: OPENROUTER_API_KEY is not configured in the .env file.")
    print("Please set it in your .env file and run again.")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

# We define a primary and fallback model in case the primary has billing/quota issues on the provider side
PRIMARY_MODEL = "deepseek/deepseek-v4-flash:free"
FALLBACK_MODEL = "liquid/lfm-2.5-1.2b-thinking:free"

def make_call(model_name):
    print(f"\n--- Calling OpenRouter with {model_name} (First reasoning call) ---")
    response = client.chat.completions.create(
      model=model_name,
      messages=[
              {
                "role": "user",
                "content": "How many r's are in the word 'strawberry'?"
              }
            ],
      extra_body={"reasoning": {"enabled": True}}
    )

    # Extract the assistant message with reasoning_details
    response_message = response.choices[0].message
    print("\nAssistant Response:")
    print(response_message.content)

    # Show reasoning details if available
    reasoning = getattr(response_message, "reasoning_details", None)
    if reasoning:
        print("\nReasoning details:")
        print(reasoning)

    # Preserve the assistant message with reasoning_details
    messages = [
      {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
      {
        "role": "assistant",
        "content": response_message.content,
        "reasoning_details": reasoning  # Pass back unmodified
      },
      {"role": "user", "content": "Are you sure? Think carefully."}
    ]

    print(f"\n--- Calling OpenRouter with {model_name} (Second continued reasoning call) ---")
    response2 = client.chat.completions.create(
      model=model_name,
      messages=messages,
      extra_body={"reasoning": {"enabled": True}}
    )

    print("\nAssistant Response 2:")
    print(response2.choices[0].message.content)

    reasoning2 = getattr(response2.choices[0].message, "reasoning_details", None)
    if reasoning2:
        print("\nReasoning details 2:")
        print(reasoning2)

try:
    make_call(PRIMARY_MODEL)
except Exception as e:
    error_str = str(e)
    # Check if this looks like a quota/credit issue or if the provider returned 402
    if "402" in error_str or "insufficient_quota" in error_str:
        print(f"\n[Warning] Primary model '{PRIMARY_MODEL}' failed due to provider credit limits (Error 402).")
        print(f"Automatically falling back to alternative free reasoning model: '{FALLBACK_MODEL}'...")
        try:
            make_call(FALLBACK_MODEL)
        except Exception as fallback_err:
            print(f"Fallback model failed too: {fallback_err}")
    else:
        print(f"\nAn error occurred during API call: {e}")
        print("Make sure you have set a valid OPENROUTER_API_KEY in the .env file.")
