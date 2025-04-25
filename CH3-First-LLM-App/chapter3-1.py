import requests
# json is a built-in module
import json

# Define the Ollama server URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Define the prompt
prompt = "What is the capital of France?"

# Define the request payload
payload = {
    "model": "llama3.1",
    "prompt": prompt,
    "stream": False
}

# Send request to the Ollama server
response = requests.post(OLLAMA_API_URL, json=payload)

# Parse response
if response.status_code == 200:
    result = response.json()
    print("Response:", result["response"])
else:
    print("Error:", response.status_code, response.text)

