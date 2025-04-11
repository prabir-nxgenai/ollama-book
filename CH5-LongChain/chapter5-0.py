from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM


# Initialize LLaMA model
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")

# SystemMessage:
#   Message for the LLM porgram - must be the first in the input message sequence
# HumanMessagse:
#   Message from a human to the AI model.
messages = [
    SystemMessage(content="Print answer in CAPS"),
    HumanMessage(content="What is the capital of UK?"),
]

print(messages)

# Invoke the model with messages (prompt)
result = llama.invoke(messages)
#print(type(messages))
print(result)

# Add to the messages list the response from the LLM, AIMessage, to increase the scope of the original prompt
messages = messages + [AIMessage(content=result)] + [HumanMessage(content="Also add the capital of France to the answer")]    
print(messages)

# Again invoke the model with expanded messages (prompt)
result = llama.invoke(messages)
# Print the final answer
print(result)
