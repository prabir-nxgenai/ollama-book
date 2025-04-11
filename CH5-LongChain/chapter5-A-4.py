from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

# Initialize the model
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")

# Define prompt template for initial feature extraction
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a nutrition expert."),
    ("human", "List the main characteristics, both good and bad, of this {product}. ")
])
# Create initial features chain
initial_chain = (
    prompt_template 
    | llama 
    | StrOutputParser() 
)
# Run  initial chain
first_result = initial_chain.invoke({"product": "Vitamin B12"})
#print("====>INITIAL CHAIN: ", first_result)

# Generate good characteristics prompt
def get_good_prompts(features):
        good_template = ChatPromptTemplate.from_messages(
             [
               ("system", "You are a nutrition expert."),
               ("human", f"Given these features, {features}, list only the good characteristics. Limit the list to 3 items."),
             ])
        return good_template.format_prompt(features=features)
 
# Define good chain for parallel execution later
good_chain = (
    RunnableLambda(lambda x: get_good_prompts(x))
    | llama 
    | StrOutputParser()
)

# Run the good chain
good_result = good_chain.invoke(first_result)
#print("****************====>GOOD CHAIN: ", good_result)

# Generate bad characteristics prompt
def get_bad_prompts(features):
        bad_template = ChatPromptTemplate.from_messages(
             [
               ("system", "You are a nutrition expert."),
               ("human", f"Given these features, {features}, list only the bad characteristics. Limit the list to 3."),
             ])
        return bad_template.format_prompt(features=features)
 
#Generate bad charcteristics prompt
# Define bad chain for  parallel execution later
bad_chain = (
    RunnableLambda(lambda x: get_bad_prompts(x))
    | llama 
    | StrOutputParser()
)

# Run the bad chain
bad_result = bad_chain.invoke(first_result)
#print("**************====>BAD CHAIN: ", bad_result)

# Combine good and the bad results
def combine_good_bad(good,bad):
     return f"Good: \n{good}\n\n:Bad:\n{bad}"

# Create good and bad parallel execution and combination chain
full_chain = (
    prompt_template 
    | llama 
    | StrOutputParser() 
    | RunnableParallel(branches={"good": good_chain, "bad": bad_chain})
    | RunnableLambda(lambda x: combine_good_bad(x["branches"]["good"], x["branches"]["bad"]))
)

# Run the chain
result = full_chain.invoke({"product": "Vitamin B12"})
print("****************====>FULL CHAIN: ", result)
