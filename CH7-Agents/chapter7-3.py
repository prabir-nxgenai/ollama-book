from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_ollama import ChatOllama
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup  # Used for Google scraping


# Define the LLaMA model
llama = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

# Define a Google web search function (No API Key)
# def google_search(query):
#     try:
#         headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
#         }
#         url = f"https://www.google.com/search?q={query}"
#         response = requests.get(url, headers=headers)

#         if response.status_code == 200:
#             soup = BeautifulSoup(response.text, "lxml")
#             results = soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd")

#             if results:
#                 return results[0].text  # Return first search result snippet
#             return "No relevant results found."
#         return f"Error: Google responded with status {response.status_code}."

#     except Exception as e:
#         return f"Error during Google search: {str(e)}"


# ---- Google Web Search (No API Key Required) ----
def google_search(query):
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        # URL encode the query
        encoded_query = quote_plus(query)
        url = f"https://www.google.com/search?q={encoded_query}"

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")

            # Try to find a relevant snippet (Google result summary)
            snippet_divs = soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd")

            # Extract the first non-empty result
            for div in snippet_divs:
                snippet = div.get_text(strip=True)
                if snippet:
                    return snippet

            return "No relevant results found."

        return f"Error: Google responded with status code {response.status_code}."

    except Exception as e:
        return f"Error during Google search: {str(e)}"



# Define a numeric calculation tool
def calculate(expression):
    try:
        return str(eval(expression, {"__builtins__": {}}))  # Ensure it returns a string
    except Exception as e:
        return f"Calculation error: {str(e)}"

# Define tools that the agent can use
tools = [
    Tool(
        name="Google Search",
        func=google_search,
        description="Searches the web for information based on the provided query."
    ),
    Tool(
        name="Calculator",
        func=calculate,  # Directly use function (no lambda needed)
        description="Evaluates mathematical expressions."
    )
]

# Initialize an agent with the tools and specify the agent type
agent = initialize_agent(
    tools=tools,
    llm=llama,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Recommended agent type
    verbose=True,  # Enables logging of the agent's reasoning
    max_iterations=3  # Avoid infinite loops
)

# Example queries
#response1 = agent.invoke({"input": "When did Neil Armstrong land on the moon?"})
#response1 = agent.invoke({"input": "Which company makes 747 aircraft?"})
response1 = agent.invoke({"input": "How many planets are there?"})
print("Response 1: ", response1)

# Debug parsing issue by returning intermediate steps
response2 = agent.invoke({"input": "What is 25 * 4 + 10?"})
print("Response 2: ", response2)
