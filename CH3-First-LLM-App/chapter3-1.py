## Import required libraries
import requests  # To handle HTTP requests to the Ollama server

# Define the base URL of the locally running Ollama API server
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Define the user input prompt (question we want to ask the LLM)
prompt = "What is the capital of France?"

# Prepare the data (payload) that will be sent in the POST request to the server
payload = {
    "model": "llama3.1",  # Specify which model to use (in this case, Llama 3.1)
    "prompt": prompt,     # The question or instruction for the model
    "stream": False       # Whether to receive a streaming response or a full response at once
}

# Send the POST request to the Ollama server with the payload as JSON
response = requests.post(OLLAMA_API_URL, json=payload)

# Check if the request was successful (HTTP status code 200 means OK)
if response.status_code == 200:
    # Parse the JSON response body
    result = response.json()

    # Print out the model's answer from the 'response' field
    print("Response:", result["response"])
else:
    # If the request failed, print out the error code and error message
    print("Error:", response.status_code, response.text)
