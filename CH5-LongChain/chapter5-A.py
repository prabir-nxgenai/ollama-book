from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

# Initialize the model
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")

# Define a prompt template
prompt = PromptTemplate.from_template("Summarize this article: {text}")

# Define an LCEL chain
chain = (
    RunnablePassthrough()  # Pass input unchanged
    | prompt  # Format input with prompt template
    | llama  # Send formatted input to the model
    | StrOutputParser()  # Parse output as a string
)

# Text to summarize- this this the starting paragrah from this chapter 
summ_txt = "Introduction LangChain is an open-source framework designed to simplify the development of applications that leverage large language models (LLMs). It provides a structured way to integrate LLMs with external data sources, tools, and reasoning capabilities, making it ideal for building advanced AI-powered applications such as chatbots, autonomous agents, and retrieval-augmented generation (RAG) systems.  The library provides a framework to easily interact with models, manage inputs, and chain different tasks together to perform multi-step workflows, thus offering powerful abstractions to make the process seamless.  It allows developers to efficiently utilize the capabilities of large language models by providing a collection of pre-built components and tools for various workflows. Head to www.langchain.com to learn more about LangChain and two other products from the company, LangGraph and LangSmith. The Docs section from the menu documents a detailed interface to the library using both Python and JavaScript.  Key FeaturesPrompt Management - LangChain helps structure and manage prompts, making it easier to create reusable, modular, and optimized interactions with LLMs.  Chains - The framework allows developers to create sequences of actions (chains) that involve multiple steps, such as calling an API, querying a database, or reasoning over data.  Memory - It supports short-term and long-term memory, enabling applications to maintain context across interactions.  Agents -  LangChain enables dynamic decision-making by allowing LLMs to choose and execute actions using external tools. Agents are such an important component that we dedicate a separate chapter to this functionality.  Retrieval & Augmentation-  It facilitates access to knowledge bases, vector stores, and databases to enhance responses with external information. We also devote an entire chapter to RAG.  Tool Integration-  Supports various integrations, including APIs, search engines, calculators, and custom functions, making applications more powerful. We look at this functionality in the chapter on Agents.  "

# Execute the chain
response = chain.invoke({"text": summ_txt})
print(response)


