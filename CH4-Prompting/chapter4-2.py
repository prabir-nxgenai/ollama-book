import requests
import json

# Define the Ollama server URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

print("Interactive Llama 3.1 Chatbot")
print("Type 'q' to exit.\n")
prompt = ""
while True:
# Take user input
    inp = input("You: ")
# Exit condition
    if inp == 'q':
        print("Exiting...")
        break
    prompt += inp + "\n"
  

    # Define the request payload
    payload = {
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False
    }

    try:
        # Send request to the Ollama server
        response = requests.post(OLLAMA_API_URL, json=payload)

        # Parse response
        if response.status_code == 200:
            result = response.json()
            print("Llama 3.1:", result.get("response", "No response received."))
        else:
            print("Error:", response.status_code, response.text)

    except requests.exceptions.RequestException as e:
        print("Error connecting to Ollama server:", e)

