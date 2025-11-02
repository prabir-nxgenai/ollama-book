## Import necessary message classes from LangChain Core
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Import the Ollama LLM (Large Language Model) interface from LangChain's Ollama module
from langchain_ollama import OllamaLLM

# Initialize the LLaMA Model
# Create an instance of the OllamaLLM object
# - Specify which model to use ("llama3.1")
# - Provide the base URL where the Ollama server is running locally
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")

# Set Up Initial Conversation Messages
# Prepare a list of messages to send to the model
# Message sequence is very important in LangChain conversation chains

messages = [
    SystemMessage(content="Print answer in CAPS"),  # Instruction to the model (sets behavior for responses)
    HumanMessage(content="What is the capital of UK?"),  # User's input/question to the model
]

# Print the list of prepared messages (for debugging/visualization)
print(messages)

# First Model Invocation
# Invoke (send) the messages to the LLaMA model
# The model will process the message history and return a response
result = llama.invoke(messages)

# Optionally check type of 'messages' list (commented out)
# print(type(messages))

# Print the model's response to the initial prompt
print(result)

# Extend the conversation by:
# - Adding the model's previous response as an AIMessage (this simulates the AI replying)
# - Adding a new HumanMessage asking to expand on the answer
messages = messages + [AIMessage(content=result)] + [HumanMessage(content="Also add the capital of France to the answer")]

# Print the updated list of messages to see the expanded conversation history
print(messages)

# Second Model Invocation
# Send the updated (longer) conversation history back to the model
# This allows the model to incorporate prior responses + new user instructions into its output
result = llama.invoke(messages)

# Print the model's final response based on the expanded conversation
print(result)

