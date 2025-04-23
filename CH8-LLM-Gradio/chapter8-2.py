# Import libraries
import httpx                # For making HTTP requests to the Ollama server
import json                 # For decoding streaming JSON lines
import gradio as gr         # For building the web UI
from langchain.memory import ConversationBufferMemory  # LangChain memory for conversation history
from langchain.schema import HumanMessage, AIMessage   # To differentiate between user and AI messages

# --- LangChain Memory Setup ---
# Initialize conversation memory that stores all messages as objects
memory = ConversationBufferMemory(return_messages=True)

def clear_question():
    #pass
     print("**************************")
     for msg in memory.chat_memory.messages:
         if isinstance(msg, HumanMessage):
             print(f"User: {msg.content}")
         elif isinstance(msg, AIMessage):
             print(f"AI: {msg.content}")

# --- Streaming Chat Function Using LangChain Memory ---
def stream_with_memory(user_input, model="llama3.1"):
    # Add the new user message to memory
    memory.chat_memory.add_user_message(user_input)

    # Reconstruct the entire conversation history to send as context
    chat_history = memory.chat_memory.messages
    history_text = ""
    for msg in chat_history:
        # Tag each message by its role
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    # Combine the history into a single prompt for the model
    prompt = history_text.strip()

    # Ollama server setup
    url = "http://localhost:11434/api/generate"  # Local Ollama API
    headers = {"Content-Type": "application/json"}  # JSON request headers
    payload = {
        "model": model,     # Model to use (llama3.1)
        "prompt": prompt,   # Full conversation history as prompt
        "stream": True      # Enable streaming of tokens
    }

    output = ""  # Accumulator for the streamed output

    # Make a streaming POST request to the Ollama server
    with httpx.stream("POST", url, headers=headers, json=payload, timeout=None) as response:
        for line in response.iter_lines():  # Iterate through streamed lines (tokens)
            if line:
                data = json.loads(line)               # Parse the JSON chunk
                token = data.get("response", "")      # Get the token from the response
                output += token                       # Append to running output
                yield output                          # Yield live updates to Gradio

    # After streaming ends, add the assistant's response to memory
    memory.chat_memory.add_ai_message(output)

# --- Gradio Web Interface ---
with gr.Blocks() as demo:
    # Title
    gr.Markdown("## Chat with LLaMA 3.1 (Ollama + LangChain Memory + Streaming)")

    with gr.Row():
        # Textbox for user prompt
        prompt_box = gr.Textbox(
            label="Your message", 
            placeholder="Ask something...", 
            lines=2
        )

        # Output box for model response (non-editable)
        output_box = gr.Textbox(
            label="Model response", 
            lines=10, 
            interactive=False
        )

    # Submit button
    submit_btn = gr.Button("Ask AI", variant="primary")
    clear_btn = gr.Button("Clear question")

    # Hook up the submit button to the streaming function
    submit_btn.click(
        fn=stream_with_memory,  # Function to call on submit
        inputs=prompt_box,      # Input to send (user prompt)
        outputs=output_box      # Output to stream to
    )


# Hook up the clear button to the prompt_box.
    clear_btn.click(
        fn=clear_question,  # Function to call on submit
        inputs=None,      # Input to send (user prompt)
        outputs=[prompt_box]      # Output to stream to
    )


# Launch the web app on localhost
demo.launch(share=True)
