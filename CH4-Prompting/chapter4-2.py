# Import necessary libraries
import requests  # For sending HTTP requests to the Ollama server

# Define the URL endpoint where the Ollama server is listening
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Print an introduction message for the user
print("Interactive Llama 3.1 Chatbot")
print("Type 'q' to exit.\n")

# Initialize the 'prompt' variable as an empty string
# This will accumulate the full conversation history (user inputs)
prompt = ""

# Start an infinite loop to allow continuous chatting
while True:
    # Prompt the user for their input
    inp = input("You: ")
    
    # Check if the user wants to exit the chat
    if inp == 'q':
        print("Exiting...")  # Notify the user that the chat is closing
        break  # Exit the while loop, thus ending the program
    
    # Append the new user input to the existing conversation
    # Add a newline after each input to keep the conversation readable for the model
    prompt += inp + "\n"

    # Prepare the data (payload) that will be sent to the Ollama server
    payload = {
        "model": "llama3.1",  # Specify the LLM model to use
        "prompt": prompt,     # Send the full accumulated conversation so far
        "stream": False       # Set to False to receive a full response at once (not chunked/streamed)
    }

    try:
        # Send the POST request to the Ollama server with the payload as JSON
        response = requests.post(OLLAMA_API_URL, json=payload)

        # If the server responds with status 200 (OK)
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()

            # Display the model's generated response
            # Use .get() to avoid KeyError if "response" key is missing
            print("Llama 3.1:", result.get("response", "No response received."))
        else:
            # If server returned an error (not status 200), print error details
            print("Error:", response.status_code, response.text)

    except requests.exceptions.RequestException as e:
        # Handle possible exceptions like connection errors, timeouts, etc.
        print("Error connecting to Ollama server:", e)

