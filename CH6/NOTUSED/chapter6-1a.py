from langchain.llms import Ollama
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.chains import LLMMathChain
from langchain.utilities import GoogleSearchAPIWrapper

# Initialize Llama3.1 LLM hosted on Ollama
llm = Ollama(model="llama3.1")

# Web Lookup Tool
search = GoogleSearchAPIWrapper()
web_lookup_tool = Tool(
    name="Web Lookup",
    func=search.run,
    description="Search the web for real-time information."
)

# Arithmetic Tool
math_chain = LLMMathChain.from_llm(llm)
arithmetic_tool = Tool(
    name="Arithmetic Calculator",
    func=math_chain.run,
    description="Performs arithmetic calculations using a lambda function."
)

# Define Multi-Agent System
agent_tools = [web_lookup_tool, arithmetic_tool]
agent = initialize_agent(
    tools=agent_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example Usage
query = "What is the population of Canada?"
response = agent.run(query)
print(response)

calculation = "What is 256 * 74?"
response = agent.run(calculation)
print(response)

