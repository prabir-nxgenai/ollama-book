## Import Required Modules
from langchain_community.chat_message_histories import ChatMessageHistory  # To store and retrieve conversation history
from langchain_core.prompts import PromptTemplate  # To create structured prompt templates
from langchain_core.runnables import RunnablePassthrough, RunnableLambda  # For building LCEL (LangChain Expression Language) chains
from langchain_core.output_parsers import StrOutputParser  # To parse model responses into clean strings
from langchain_ollama import OllamaLLM  # Interface to interact with the locally running Ollama server
from langchain_core.messages import HumanMessage, AIMessage  # Message types


# Initialize the LLM Model
# Create an instance of the LLaMA model (Llama 3.1) hosted locally via Ollama
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")


# Set Up Conversation Memory 
# Initialize a ChatMessageHistory object to keep track of the conversation messages
chat_history = ChatMessageHistory()


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
    # Retrieve all previous messages from chat history
    history = chat_history.messages

    # Format each message like "Human: message_content" or "Ai: message_content"
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
user_input_1 = "What is the capital of France?"
response_1 = chain.invoke({"input": user_input_1})
print(response_1)

# Save the conversation to chat history
chat_history.add_message(HumanMessage(content=user_input_1))
chat_history.add_message(AIMessage(content=response_1))


# Step 2: Ask a question about India
user_input_2 = "Btw, can you also provide the name of the capital of India?"
response_2 = chain.invoke({"input": user_input_2})
print(response_2)

# Save the conversation to chat history
chat_history.add_message(HumanMessage(content=user_input_2))
chat_history.add_message(AIMessage(content=response_2))


# Step 3: Request to show the last two answers line by line
user_input_3 = "Show the last two answers in a new line."
response_3 = chain.invoke({"input": user_input_3})
print(response_3)

# Save the conversation to chat history
chat_history.add_message(HumanMessage(content=user_input_3))
chat_history.add_message(AIMessage(content=response_3))


# Step 4: Request to repeat the answers but in all capital letters
user_input_4 = "Please provide the answers again in CAPS."
response_4 = chain.invoke({"input": user_input_4})
print(response_4)

# Save the conversation to chat history
chat_history.add_message(HumanMessage(content=user_input_4))
chat_history.add_message(AIMessage(content=response_4))
