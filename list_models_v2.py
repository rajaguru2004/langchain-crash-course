
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

print("Loading .env...", flush=True)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key loaded: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}", flush=True)

if not api_key:
    print("Error: GOOGLE_API_KEY not found.", flush=True)
    sys.exit(1)

genai.configure(api_key=api_key)
print("Configured genai with API key.", flush=True)

try:
    print("Listing models...", flush=True)
    count = 0
    with open("available_models.txt", "w") as f:
        for m in genai.list_models():
            count += 1
            print(f"Found model: {m.name}", flush=True)
            f.write(f"{m.name}\n")
    print(f"Total models found: {count}", flush=True)
except Exception as e:
    print(f"Error listing models: {e}", flush=True)

print("\nTesting generation with gemini-1.5-flash...", flush=True)
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello, can you hear me?")
    print(f"Generation success: {response.text}", flush=True)
except Exception as e:
    print(f"Error generating content: {e}", flush=True)
