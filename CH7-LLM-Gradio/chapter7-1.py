import requests
import json
import gradio as gr

# Define the Ollama server URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def query_ollama(prompt, model="llama3.1"):
    """Send a prompt to the Ollama LLM server and return the response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    # Send request to Ollama server
    response = requests.post(OLLAMA_API_URL, json=payload)

    # Parse and return response
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "No response received.")
    else:
        return f"Error {response.status_code}: {response.text}"
    
# Gradio UI
demo = gr.Interface(
    fn=query_ollama,
    inputs=[
        gr.Textbox(label="Enter your prompt", placeholder="Ask me anything..."),
        gr.Textbox(label="Model Name", value="llama3.1", interactive=True)
    ],
    outputs="text",
    title="Ollama LLM Chat",
    description="Enter a prompt and get responses from the Ollama LLM server.",
    allow_flagging="never"
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=True)

