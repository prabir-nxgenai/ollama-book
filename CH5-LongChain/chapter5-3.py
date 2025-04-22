from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Initialize the model
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")


# Define a prompt template
prompt = PromptTemplate.from_template("What is capital of {country}?")

# We declare a Python Lambda function capitalize_output that takes the 
# output from the previous step in the chain and capitalizes it.
capitalize_output = RunnableLambda(lambda x: x.upper())

# Define an LCEL chain
chain = (
    RunnablePassthrough()
    | prompt  # Format input with prompt template
    | llama  # Send formatted input to the model
    | StrOutputParser()  # Parse output as a string
    | capitalize_output
)

# Execute the chain
response = chain.invoke({"country": "France"})
print( response)


