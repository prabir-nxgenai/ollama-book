## Import Required Modules
import requests  # For sending HTTP POST requests to the Ollama server
import gradio as gr  # For building the web-based UI (Gradio)


# Define the Ollama Server Endpoint 
# URL where the local Ollama LLM server is running
OLLAMA_API_URL = "http://localhost:11434/api/generate"


# Define the Query Function to Interact with the LLM 
def query_ollama(prompt, model="llama3.1"):
    """
    Sends a prompt to the Ollama LLM server and returns the generated response.

    Args:
        prompt (str): The user's input prompt.
        model (str): The model name to query (default is "llama3.1").

    Returns:
        str: The LLM's response text or an error message.
    """

    # Prepare the payload (data) to send in the POST request
    payload = {
        "model": model,    # Specify which model to use
        "prompt": prompt,  # Pass the user's prompt
        "stream": False    # Set to False to receive the full response in one go
    }

    # Send the POST request to the Ollama server
    response = requests.post(OLLAMA_API_URL, json=payload)

    # Parse and handle the server response
    if response.status_code == 200:
        # If successful, return the model's response text
        result = response.json()
        return result.get("response", "No response received.")
    else:
        # If the request failed, return a descriptive error message
        return f"Error {response.status_code}: {response.text}"


# Create the Gradio UI Interface 
# Set up a Gradio Interface to make it easy to interact with the LLM via a web page
demo = gr.Interface(
    fn=query_ollama,  # Function that runs when the user submits input
    inputs=[
        gr.Textbox(label="Enter your prompt", placeholder="Ask me anything..."),  # User input for the prompt
        gr.Textbox(label="Model Name", value="llama3.1", interactive=True)  # Optional input to specify model
    ],
    outputs="text",  # Output type: plain text
    title="Ollama LLM Chat",  # Title displayed at the top of the app
    description="Enter a prompt and get responses from the Ollama LLM server.",  # Short description
    flagging_mode="never"  # Disable example flagging feature in Gradio
)


# Launch the Gradio App
# Start the Gradio app only if this script is run directly (not imported elsewhere)
if __name__ == "__main__":
    demo.launch(share=True)  # Launch the app and optionally share it with a public link

