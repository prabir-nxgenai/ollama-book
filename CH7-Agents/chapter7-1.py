## Import Required Modules
from langchain_core.tools import tool  # To define functions as "tools" that an agent can use
from langchain_ollama import ChatOllama  # To interact with LLaMA models served by Ollama
from ddgs import DDGS  # To perform DuckDuckGo search queries
from langchain.agents import create_agent

# Define a Function to Get the Current Time
@tool
def get_time(query: str = "") -> str:
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
@tool
def duckduckgo_search(query: str = "") -> str:
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


TOOLS=[get_time, duckduckgo_search]


# Initialize the LLaMA Model via Ollama
# Create a ChatOllama object pointing to a local LLaMA 3.1 model instance
llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")



# Initialize the LangChain Agent
agent = create_agent(llm, TOOLS)


# Execute Queries Using the Agent
# Query 1: Ask the agent to fetch the current time
print(agent.invoke({"messages": [{"role": "user", "content": "What is the current time? Provide answer in YYYY-MM-DD HH:MM:SS format"}]}))
print ("###############################################")
print ("###############################################")
# Query 2: Ask the agent to search the Internet for a fact
print(agent.invoke({"messages": [{"role": "user", "content": "What is the closest planet to the sun?"}]}))

