# Import the ollama library
# This library allows us to interact with a local Ollama server to query language models like Llama 3.1
import ollama

# Define a function to query the Ollama server with a user prompt
def query_ollama(prompt, model="llama3.1"):
    """
    Sends a prompt to the specified Ollama model and returns the response.

    Args:
        prompt (str): The user's input/question to send to the model.
        model (str): The model to query (default is 'llama3.1').

    Returns:
        dict: The full response from the Ollama server, including the model's output.
    """
    # Call the Ollama server's chat function
    # 'model' specifies which language model to use
    # 'messages' contains a list of conversational turns (in this case, a single user message)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]  # Format: Role (user), Content (the prompt text)
    )
    
    # Return the entire server response (a dictionary containing model's answer and other metadata)
    return response

# Test the function by asking "What is the capital of France?"
# The output will be the model's response in a dictionary format
print(query_ollama("What is the capital of France?"))

