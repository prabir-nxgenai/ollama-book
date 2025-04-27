# Import necessary modules from LangChain and LangChain-Ollama
from langchain_ollama import OllamaLLM  # Interface for calling a local Ollama LLM server
from langchain.prompts import ChatPromptTemplate  # For defining structured chat-style prompts
from langchain_core.output_parsers import StrOutputParser  # To parse model outputs as plain strings
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough  # Tools for building LCEL chains

# Initialize the LLaMA Model 
# Create an instance of the LLaMA model (Llama 3.1)
# - 'base_url' points to the local Ollama server
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")


# Define Initial Prompt Template 
# Define a chat prompt that sets the system role as a nutrition expert
# Human input requests listing good and bad features of a given product
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a nutrition expert."),  # System message to set model's role
    ("human", "List the main characteristics, both good and bad, of this {product}.")  # User instruction with a placeholder
])

# Define the Initial Feature Extraction Chain 
# Create the first chain:
# 1. Format the input product name using the prompt template
# 2. Send the formatted prompt to the LLaMA model
# 3. Parse the output as a plain string
initial_chain = (
    prompt_template 
    | llama 
    | StrOutputParser() 
)

# Run the initial chain by providing a specific product ("Vitamin B12")
# The result will contain all (good + bad) characteristics
first_result = initial_chain.invoke({"product": "Vitamin B12"})


# Define Good Characteristics Extraction Chain 
# Define a helper function to generate a prompt focusing only on good features
def get_good_prompts(features):
    good_template = ChatPromptTemplate.from_messages([
        ("system", "You are a nutrition expert."),
        ("human", f"Given these features, {features}, list only the good characteristics. Limit the list to 3 items."),
    ])
    return good_template.format_prompt(features=features)

# Define a chain that:
# 1. Takes the initial features
# 2. Formats a "good characteristics" prompt
# 3. Sends it to the LLM
# 4. Parses the output
good_chain = (
    RunnableLambda(lambda x: get_good_prompts(x))
    | llama 
    | StrOutputParser()
)

# Execute the good features extraction separately (optional step)
good_result = good_chain.invoke(first_result)


# Define Bad Characteristics Extraction Chain
# Define a helper function to generate a prompt focusing only on bad features
def get_bad_prompts(features):
    bad_template = ChatPromptTemplate.from_messages([
        ("system", "You are a nutrition expert."),
        ("human", f"Given these features, {features}, list only the bad characteristics. Limit the list to 3."),
    ])
    return bad_template.format_prompt(features=features)

# Define a chain that:
# 1. Takes the initial features
# 2. Formats a "bad characteristics" prompt
# 3. Sends it to the LLM
# 4. Parses the output
bad_chain = (
    RunnableLambda(lambda x: get_bad_prompts(x))
    | llama 
    | StrOutputParser()
)

# Execute the bad features extraction separately (optional step)
bad_result = bad_chain.invoke(first_result)


# Define Combination Function 
# Helper function to combine good and bad characteristics into a formatted string
def combine_good_bad(good, bad):
    return f"Good: \n{good}\n\nBad:\n{bad}"


# Define the Full Parallel Chain 
# Full chain overview:
# 1. Format the initial product prompt
# 2. Send to the LLaMA model to get all features
# 3. Parse initial output
# 4. Split the output into two branches ("good" and "bad") processed in parallel
# 5. Combine the results into a single string

full_chain = (
    prompt_template  # Format the initial product prompt
    | llama           # Query the model
    | StrOutputParser()  # Parse model response
    | RunnableParallel(branches={"good": good_chain, "bad": bad_chain})  # Run good and bad extractions in parallel
    | RunnableLambda(lambda x: combine_good_bad(x["branches"]["good"], x["branches"]["bad"]))  # Combine the results
)

# Execute the Full Chain 
# Provide the initial input ("Vitamin B12")
# - The full chain extracts, separates, and combines good and bad features
result = full_chain.invoke({"product": "Vitamin B12"})

# Print the final combined output
print("****************====>FULL CHAIN: ", result)

