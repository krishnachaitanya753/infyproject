import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Load env variables
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
if not api_key or api_key == "your_nvidia_api_key_here":
    print("Error: NVIDIA_API_KEY is not configured in the .env file.")
    print("Please set your NVIDIA_API_KEY in the .env file and run again.")
    sys.exit(1)

client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=api_key
)

# Active model on Nvidia build
MODEL = "minimaxai/minimax-m2.7"

SYSTEM_PROMPT = """You are a video script generator. Your job is to translate a short one-line story concept into a highly structured, low-level script suitable for video generation/production.

Follow this exact formatting style:

SCENE 1 - [SCENE TITLE]

# VISUAL: [Detailed description of the setting, camera positioning, and character placement]

ACTION: [Action description of the characters]

[CHARACTER] SPEAKS:
"[Direct spoken dialogue]"

Use these conventions:
- Use all caps for markers like "SCENE X - [TITLE]", "# VISUAL:", "ACTION:", and "[CHARACTER] SPEAKS:".
- Write dialogue that sounds highly natural, casual, and colloquial. Use conversational fillers (e.g., "So glad you're here", "Okay so -", "honestly", "Not bad for...").
- Include physical gestures in ACTION lines (e.g., "G1 points toward the engine", "G1 turns and walks", "G1 spreads both arms open").
- Conclude the entire script with:
## END OF STORY

Here is an example structure:
=========================================
SCENE 1 - INTRODUCTION

# VISUAL: The entrance of Infosys Living Labs. A person walks through the door. G1 is already inside, standing near the entrance.

ACTION: G1 brings both hands together and performs a Namaste bow toward the visitor.

G1 SPEAKS:
"Hey - Namaste! Welcome. I'm G1, and I'll be showing you around today."

ACTION: G1 straightens up and extends one arm out in a casual welcoming gesture toward the lab.

G1 SPEAKS:
"So glad you're here. Let's dive in."

## END OF STORY
========================================="""

def generate_script(prompt: str):
    print(f"\nGenerating script for: '{prompt}'...")
    print("-" * 50)
    
    try:
        # Use streaming to output in real-time
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate a video script for this concept: {prompt}"}
            ],
            temperature=1,
            top_p=0.95,
            max_tokens=8192,
            stream=True
        )
        
        script_content = []
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            content = chunk.choices[0].delta.content
            if content is not None:
                print(content, end="", flush=True)
                script_content.append(content)
        print()
        
        # Write to file
        output_file = "generated_script.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(script_content))
        print("-" * 50)
        print(f"Saved generated script to [generated_script.md](file:///c:/Users/Krishna/Desktop/infyproject/{output_file})")
        
    except Exception as e:
        print(f"\nAn error occurred during generation: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        story_prompt = " ".join(sys.argv[1:])
    else:
        story_prompt = input("Enter a one-line story concept: ")
        
    if not story_prompt.strip():
        print("Error: Concept cannot be empty.")
        sys.exit(1)
        
    generate_script(story_prompt)
