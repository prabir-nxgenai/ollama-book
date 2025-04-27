# Import Required Libraries
import httpx  # For making asynchronous and streaming HTTP requests to the Ollama server
import json  # For decoding streaming JSON data (Newline-delimited JSON - NDJSON)
import gradio as gr  # For building the web user interface (UI)
from langchain.memory import ConversationBufferMemory  # To store conversation history during the chat
from langchain.schema import HumanMessage, AIMessage  # To represent user (Human) and AI (Assistant) messages


# Set Up LangChain Memory
# Initialize conversation memory
# - Stores all the exchanged messages (user and AI) to maintain context
# - Messages are stored as structured objects (HumanMessage, AIMessage)
memory = ConversationBufferMemory(return_messages=True)


# Define a Clear Question Function --------------------------------
def clear_question():
    """
    Placeholder function tied to the 'Clear' button in Gradio UI.
    (Currently does nothing but can be expanded if needed.)
    """
    pass
    # # (Optional for Debugging) Print the chat history
    # for msg in memory.chat_memory.messages:
    #     if isinstance(msg, HumanMessage):
    #         print(f"User: {msg.content}")
    #     elif isinstance(msg, AIMessage):
    #         print(f"AI: {msg.content}")


# Define the Streaming Chat Function --------------------------------

def stream_with_memory(user_input, model="llama3.1"):
    """
    Handles interaction with Ollama server.
    - Sends full conversation history as prompt
    - Receives and streams the model's response token-by-token
    - Updates memory after AI response
    """
    # Add the new user input to memory
    memory.chat_memory.add_user_message(user_input)

    # Reconstruct conversation history (User: ... Assistant: ...) for context
    chat_history = memory.chat_memory.messages
    history_text = ""
    for msg in chat_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    # Build the full prompt by combining history
    prompt = history_text.strip()

    # Setup request parameters for Ollama
    url = "http://localhost:11434/api/generate"  # Local API endpoint
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,   # Specify which model to use
        "prompt": prompt,  # Send the entire conversation history
        "stream": True    # Enable token-by-token streaming response
    }

    output = ""  # Initialize output accumulator

    # Make a streaming POST request to Ollama server
    with httpx.stream("POST", url, headers=headers, json=payload, timeout=None) as response:
        for line in response.iter_lines():  # Iterate over streamed tokens
            if line:
                data = json.loads(line)  # Parse each JSON line
                token = data.get("response", "")  # Extract the token from server response
                output += token  # Append the token to the running output
                yield output  # Yield (stream) the partial output to Gradio

    # After streaming is complete, save the assistant's full response into memory
    memory.chat_memory.add_ai_message(output)


# Build the Gradio Web Interface
with gr.Blocks() as demo:
    # Title of the web app
    gr.Markdown("## Chat with LLaMA 3.1 (Ollama + LangChain Memory + Streaming)")

    with gr.Row():
        # Textbox for user to type their message
        prompt_box = gr.Textbox(
            label="Your message",
            placeholder="Ask something...", 
            lines=2
        )

        # Textbox for displaying model's streamed response (non-editable)
        output_box = gr.Textbox(
            label="Model response",
            lines=10,
            interactive=False  # Make it read-only
        )

    # Buttons for submitting a query and clearing the input
    submit_btn = gr.Button("Ask AI", variant="primary")
    clear_btn = gr.Button("Clear question")

    # Connect the Submit button to the streaming chat function
    submit_btn.click(
        fn=stream_with_memory,  # Function to execute on button click
        inputs=prompt_box,       # Input taken from user textbox
        outputs=output_box       # Output displayed in output textbox
    )

    # Connect the Clear button to the clear function
    clear_btn.click(
        fn=clear_question,  # Clear function (currently a placeholder)
        inputs=None,         # No input needed
        outputs=[prompt_box]  # Clear the prompt textbox
    )


# Launch the Web Application
# Start the Gradio app if the script is run directly
if __name__ == "__main__":
    demo.launch(share=True)  # 'share=True' creates a publicly accessible URL (optional)

