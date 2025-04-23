import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_ollama import OllamaLLM
from duckduckgo_search import DDGS

# ---- DuckDuckGo Search Tool (No API Key) ----
# def duckduckgo_search(query: str) -> str:
#     try:
#         headers = {
#             "User-Agent": (
#                 "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                 "AppleWebKit/537.36 (KHTML, like Gecko) "
#                 "Chrome/91.0.4472.124 Safari/537.36"
#             )
#         }
#         encoded_query = quote_plus(query)
#         url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
#         response = requests.get(url, headers=headers)

#         if response.status_code == 200:
#             soup = BeautifulSoup(response.text, "html.parser")
#             results = soup.find_all("a")

#             for result in results:
#                 snippet = result.get_text(strip=True)
#                 if snippet and "more results" not in snippet.lower():
#                     return snippet

#             return "No relevant results found."

#         return f"DuckDuckGo returned status code {response.status_code}."

#     except Exception as e:
#         return f"Error during DuckDuckGo search: {str(e)}"


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


# ---- Wrap as LangChain Tool ----
duck_tool = Tool(
    name="duckduckgo_search",
    func=duckduckgo_search,
    description="Use this tool to answer real-world factual questions by searching DuckDuckGo. Input should be a search query string."
)

# ---- Set Up LLaMA 3.1 via Ollama ----
llm = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")

# ---- Initialize LangChain Agent ----
agent_executor = initialize_agent(
    tools=[duck_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ---- Ask the Question ----
question = "How many planets are in our solar system?"
response = agent_executor.invoke(question)

# ---- Output the Answer ----
print("\nFinal Answer:", response)

