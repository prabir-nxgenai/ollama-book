from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
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

# Define tools that the agent can use
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

# Initialize an agent with the tools and specify the agent type
agent = initialize_agent(
    tools=tools,
    llm=llama,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Recommended agent type
    verbose=True  # Optional: Enables logging of the agent's reasoning
)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)
# Example queries
#response1 = agent.run("What is the latest news about LangChain?")
#print(response1)

response2 = executor.invoke({"input": "What is 25 * 4 + 10?"})
print(response2)
