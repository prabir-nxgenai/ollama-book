from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# Initialize the model (Ensure "llama3.1" is the correct model name in Ollama)
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")

# Initialize memory with ChatMessageHistory
chat_history = ChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=chat_history, return_messages=True)

# Function to format chat history correctly for the prompt
def format_chat_history(input_data):
    history = memory.chat_memory.messages
    formatted_history = "\n".join(
        [f"{msg.type.capitalize()}: {msg.content}" for msg in history]
    )
    return {"chat_history": formatted_history, "input": input_data["input"]}

# Define prompt template
prompt = PromptTemplate.from_template("{chat_history}\nHuman: {input}\nAssistant:")

# Create chain using RunnableSequence
chain = (
    RunnablePassthrough()
    | RunnableLambda(format_chat_history)  # Process chat history
    | prompt
    | llama
    | StrOutputParser()
)

# Simulate a conversation
print(chain.invoke({"input": "What is the capital of France?"}))
memory.save_context({"input": "What is the capital of France?"}, {"output": "The capital of France is Paris."})

print(chain.invoke({"input": "Btw, can you also provide the name of the capital of India?"}))
memory.save_context({"input": "Btw, can you also provide the name of the capital of India?"}, {"output": "The capital of India is New Delhi."})

print(chain.invoke({"input": "Show the last two answers in a new line."}))
print(chain.invoke({"input": "Please provide the answers again in CAPS."}))
