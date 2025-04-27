# Import Required Modules
from langchain_community.chat_message_histories import ChatMessageHistory  # To store and retrieve conversation history
from langchain.memory import ConversationBufferMemory  # A memory class to buffer the conversation context

from langchain_core.prompts import PromptTemplate  # To create structured prompt templates
from langchain_core.runnables import RunnablePassthrough, RunnableLambda  # For building LCEL (LangChain Expression Language) chains
from langchain_core.output_parsers import StrOutputParser  # To parse model responses into clean strings
from langchain_ollama import OllamaLLM  # Interface to interact with the locally running Ollama server


# Initialize the LLM Model
# Create an instance of the LLaMA model (Llama 3.1) hosted locally via Ollama
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")


# Set Up Conversation Memory 
# Initialize a ChatMessageHistory object to keep track of the conversation messages
chat_history = ChatMessageHistory()

# Initialize a memory buffer that uses ChatMessageHistory
# - 'return_messages=True' means when retrieving memory, the messages will be structured
memory = ConversationBufferMemory(chat_memory=chat_history, return_messages=True)


# Define Helper Function to Format Chat History 
# Define a function that formats the stored chat history for input into the prompt
def format_chat_history(input_data):
    """
    Formats the conversation history into a readable text block
    and returns it along with the new user input.
    
    Args:
        input_data (dict): Contains the new user input.
    
    Returns:
        dict: A dictionary with 'chat_history' and 'input' keys.
    """
    history = memory.chat_memory.messages  # Retrieve all previous messages

    # Format each message like "Human: message_content" or "AI: message_content"
    formatted_history = "\n".join(
        [f"{msg.type.capitalize()}: {msg.content}" for msg in history]
    )

    # Return a dictionary expected by the prompt template
    return {"chat_history": formatted_history, "input": input_data["input"]}


# Define the Prompt Template
# Create a prompt template that structures the chat history and the current user input
# '{chat_history}' will be replaced by the formatted past conversation
# '{input}' will be replaced by the current user input
prompt = PromptTemplate.from_template("{chat_history}\nHuman: {input}\nAssistant:")


# Build the LCEL Chain
# Build the chain by linking components together:
chain = (
    RunnablePassthrough()  # Pass input directly into the chain without modification
    | RunnableLambda(format_chat_history)  # Apply the format_chat_history function to structure history + input
    | prompt  # Format it using the PromptTemplate
    | llama  # Send the formatted prompt to the LLaMA model
    | StrOutputParser()  # Parse the raw model response into a plain string
)


# Simulate a Conversation
# Step 1: Ask a question about France
print(chain.invoke({"input": "What is the capital of France?"}))
# Save the context (input and output) to memory for future conversation
memory.save_context(
    {"input": "What is the capital of France?"},
    {"output": "The capital of France is Paris."}
)

# Step 2: Ask a question about India
print(chain.invoke({"input": "Btw, can you also provide the name of the capital of India?"}))
# Save this conversation turn to memory
memory.save_context(
    {"input": "Btw, can you also provide the name of the capital of India?"},
    {"output": "The capital of India is New Delhi."}
)

# Step 3: Request to show the last two answers line by line
print(chain.invoke({"input": "Show the last two answers in a new line."}))

# Step 4: Request to repeat the answers but in all capital letters
print(chain.invoke({"input": "Please provide the answers again in CAPS."}))

