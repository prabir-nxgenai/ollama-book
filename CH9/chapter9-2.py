import requests
import tkinter as tk
from tkinter import scrolledtext

# Define the Ollama server URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def send_message():
    """Handles sending a message to the Ollama server and displaying the response."""
    user_input = entry.get().strip()
    if not user_input:
        return

    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, f"\nYou: {user_input}\n", "user")
    chat_display.config(state=tk.DISABLED)

    # Define the request payload
    payload = {
        "model": model_var.get(),
        "prompt": user_input,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            bot_reply = result.get("response", "No response received.")
        else:
            bot_reply = f"Error {response.status_code}: {response.text}"

    except requests.exceptions.RequestException as e:
        bot_reply = f"❌ Network error: {e}"

    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, f"🤖 Ollama: {bot_reply}\n", "bot")
    chat_display.config(state=tk.DISABLED)

    entry.delete(0, tk.END)  # Clear input field
    chat_display.yview(tk.END)  # Auto-scroll to the latest message

# Create the main application window
root = tk.Tk()
root.title("Ollama AI Chatbot")

# Chat history display
chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20)
chat_display.pack(padx=10, pady=10)
chat_display.config(state=tk.DISABLED)  # Make it read-only

# Model selection
model_var = tk.StringVar(value="llama3.1")
model_entry = tk.Entry(root, textvariable=model_var, width=20)
model_entry.pack(pady=5)
#model_entry.insert(0, "llama3.1")  # Default model

# User input field
entry = tk.Entry(root, width=50)
entry.pack(pady=5)

# Send button
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(pady=5)

# Run the application
root.mainloop()

