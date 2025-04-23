from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain_ollama import OllamaLLM

# Define a simple function
def get_time(_):
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Register function as a tool
time_tool = Tool(name="current_time", func=get_time, description="Returns the current time")

# Initialize an agent
llm = OllamaLLM(model="llama3.1")
agent = initialize_agent([time_tool], llm, agent="zero-shot-react-description", handle_parsing_errors=True, verbose=True)


# Execute a query that triggers the function
print(agent.invoke("What is the current time?"))

