from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOllama
from duckduckgo_search import DDGS

# Define a function to get the current time
def get_time(_):
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#Define a function to search the internet
def duckduckgo_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            for result in results:
                snippet = result.get("body") or result.get("snippet")
                if snippet:
                    return snippet
            return "No relevant results found."
    except Exception as e:
        return f"Error during DuckDuckGo search: {str(e)}"
    
# Register the two functions as a tools
time_tool = Tool(
        name="current_time", 
        func=get_time, 
        description="Returns the current time"
)
duck_tool = Tool(
    name="duckduckgo_search",
    func=duckduckgo_search,
    description="Searches the Internet. Input should be a search query string."
)
# Initialize the Ollama model
llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

# Initialize the agent
agent = initialize_agent([time_tool, duck_tool], llm, agent="zero-shot-react-description", handle_parsing_errors=True, verbose=True)


# Execute queries that trigger time and search tools by the agent
print(agent.invoke("Question 1: What is the current time? Provide answer in YYYY-MM-DD HH:MM:SS format"))
print(agent.invoke("Question 2: What is the closest planet to the sun?"))

