## Import necessary classes and functions from LangChain and LangChain-Ollama

from langchain_ollama import OllamaLLM  # Interface to communicate with the Ollama LLM server
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate  # Tools for creating text prompt templates
from langchain_core.output_parsers import StrOutputParser  # Parses model responses into plain strings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda  # Tools for building LCEL chains

# Initialize the LLaMA Model 
# Create an instance of the LLaMA model from Ollama
# - Specify the model name ("llama3.1")
# - Set the base URL of the locally running Ollama server
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")


# Define the Prompt Template 
# Create a structured PromptTemplate
# The template defines a simple question, where {country} will be dynamically replaced
prompt = PromptTemplate.from_template("What is capital of {country}?")


# Define a Lambda Function to Post-process Output 
# Define a RunnableLambda that modifies the output from the LLM
# - The lambda function takes a string 'x' and converts it to all UPPERCASE letters
# - This step post-processes the LLM output before final presentation
capitalize_output = RunnableLambda(lambda x: x.upper())


# Build the LCEL Chain 
# LCEL (LangChain Expression Language) Chain:
# This defines a sequence of operations, linked together using the '|' operator

chain = (
    RunnablePassthrough()  # 1. Initially passes input unchanged (raw dictionary {"country": "France"})
    | prompt               # 2. Formats the input into a full prompt string using the template
    | llama                # 3. Sends the formatted prompt to the LLaMA model and retrieves a response
    | StrOutputParser()    # 4. Parses the raw model output into a simple string
    | capitalize_output    # 5. Applies the lambda function to capitalize the final response
)


# Execute the Chain 
# Provide the input value ("France") mapped to the variable {country}
response = chain.invoke({"country": "France"})

# Print the final processed output (which will be capitalized)
print(response)

