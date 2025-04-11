from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough

# Initialize the model
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")


# No LCEL
# Define a prompt template using  unstructured prompt 
template = "What is the capital of {country}?"
prompt_template = ChatPromptTemplate.from_template(template)

#Create the prompt that can be passed to the LLM. Note the use of .format() 
#function call instead of calling the template string directly
prompt = prompt_template.format(country="France")

# Invoke the model with the formatted prompt
result = llama.invoke(prompt)

# Print the response content
print("No LCEL: ", result)

# Using LCEL
# Define a prompt template
prompt = PromptTemplate.from_template("What is capital of {country}?")

# Define an LCEL chain
chain = (
    RunnablePassthrough()
    | prompt  # Format input with prompt template
    | llama  # Send formatted input to the model
    | StrOutputParser()  # Parse output as a string
)

# Execute the chain
response = chain.invoke({"country": "France"})
print("Using LCEL: ", response)


