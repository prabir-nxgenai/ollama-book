# Import required libraries
import requests  # For sending HTTP requests to the Ollama server

# Define the URL endpoint for the local Ollama server's generate API
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Print welcome message to the user
print("Interactive Llama 3.1 Chatbot")
print("Type 'q' to exit.\n")  # Provide instructions on how to exit

# Start an infinite loop to allow continuous interaction with the chatbot
while True:
    # Prompt the user for input
    prompt = input("You: ")
    
    # Check if the user wants to quit
    if prompt.lower() == 'q':
        print("Exiting...")  # Notify that the program is exiting
        break  # Break the loop to end the program

    # Prepare the payload (data) to send in the HTTP POST request
    payload = {
        "model": "llama3.1",  # Specify the model name to use
        "prompt": prompt,     # Pass the user's message as the prompt
        "stream": False       # Set to False to receive a full response (not streamed in parts)
    }

    try:
        # Send the POST request to the Ollama server with the payload formatted as JSON
        response = requests.post(OLLAMA_API_URL, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # If successful, parse the JSON response
            result = response.json()
            # Display the model's response to the user
            print("Llama 3.1:", result.get("response", "No response received."))
        else:
            # If the server returns an error status, display the error code and message
            print("Error:", response.status_code, response.text)

    except requests.exceptions.RequestException as e:
        # Handle exceptions such as server connection errors
        print("Error connecting to Ollama server:", e)

