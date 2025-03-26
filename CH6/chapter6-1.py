from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentType, AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
import requests

# Define the LLaMA model
llama = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

# Define a proper web search tool function
def search_tool(query):
    try:
        response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
        data = response.json()
        return data.get("AbstractText", "No relevant information found.")
    except Exception as e:
        return f"Error during web search: {str(e)}"

# Define a numeric calculation tool
def calculate(expression):
    try:
        return eval(expression, {"__builtins__": {}})  # Restrict built-in functions
    except Exception as e:
        return f"Calculation error: {str(e)}"

# Define tools
tools = [
    Tool(
        name="Search Tool",
        func=search_tool,
        description="Searches the web for information based on the provided query."
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(calculate(x)),
        description="Evaluates mathematical expressions."
    )
]

# Create a ReAct agent (updated approach)
agent = create_react_agent(llm=llama, tools=tools)

# Use `from_agent_and_tools` to create the executor
executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=3)

# Run the agent
response2 = executor.invoke({"input": "What is 25 * 4 + 10?"})
print(response2)
