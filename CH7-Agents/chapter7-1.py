# Import Required Modules
from langchain.tools import Tool  # To define functions as "tools" that an agent can use
from langchain.agents import initialize_agent  # To create and configure an LLM agent
from langchain_community.chat_models import ChatOllama  # To interact with LLaMA models served by Ollama
from duckduckgo_search import DDGS  # To perform DuckDuckGo search queries


# Define a Function to Get the Current Time
def get_time(_):
    """
    Returns the current system time formatted as YYYY-MM-DD HH:MM:SS.
    
    Args:
        _ (any): Placeholder argument (ignored).
    
    Returns:
        str: Formatted current date and time.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Define a Function to Perform Internet Search
def duckduckgo_search(query: str) -> str:
    """
    Performs a DuckDuckGo text search and returns the first relevant snippet.

    Args:
        query (str): The search query string.
    
    Returns:
        str: The first found snippet or an error message.
    """
    try:
        with DDGS() as ddgs:  # Create a DuckDuckGo Search client
            results = ddgs.text(query, max_results=3)  # Get up to 3 search results
            for result in results:
                snippet = result.get("body") or result.get("snippet")  # Try to find a text snippet
                if snippet:
                    return snippet  # Return the first non-empty snippet
            return "No relevant results found."  # Fallback if no snippets found
    except Exception as e:
        return f"Error during DuckDuckGo search: {str(e)}"  # Return error message if search fails


# Register the Functions as Tools
# Wrap the get_time function into a LangChain Tool
time_tool = Tool(
    name="current_time",
    func=get_time,
    description="Returns the current time"
)

# Wrap the duckduckgo_search function into another Tool
duck_tool = Tool(
    name="duckduckgo_search",
    func=duckduckgo_search,
    description="Searches the Internet. Input should be a search query string."
)


# Initialize the LLaMA Model via Ollama
# Create a ChatOllama object pointing to a local LLaMA 3.1 model instance
llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")


# Initialize the LangChain Agent
# Create an agent capable of deciding when to use the tools (zero-shot reasoning)
# - [time_tool, duck_tool] = available tools for the agent
# - llm = model the agent will use to reason and decide actions
# - agent = "zero-shot-react-description" strategy (decides tools based on input description)
# - handle_parsing_errors = True ensures robustness if outputs are malformed
# - verbose = True for detailed printout of internal decision steps
agent = initialize_agent(
    tools=[time_tool, duck_tool],
    llm=llm,
    agent="zero-shot-react-description",
    handle_parsing_errors=True,
    verbose=True
)


# Execute Queries Using the Agent
# Query 1: Ask the agent to fetch the current time
print(agent.invoke("Question 1: What is the current time? Provide answer in YYYY-MM-DD HH:MM:SS format"))

# Query 2: Ask the agent to search the Internet for a fact
print(agent.invoke("Question 2: What is the closest planet to the sun?"))

