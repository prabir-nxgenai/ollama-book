from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

# Initialize the model
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")

# Define prompt template for initial feature extraction
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a nutrition expert."),
    ("human", "List the main characteristics, both good and bad, of this {product}. Limit the list to 3 items per category.")
])
# Create initial features chain
chain_1 = (
    prompt_template 
    | llama 
    | StrOutputParser() 
)
# Run  chain_1
result_1 = chain_1.invoke({"product": "Vitamin B12"})
print("====>Type for result_1: ", type(result_1))
print("====>CHAIN 1: ", result_1)
# Create initial features chain with prompt as the output
chain_1_1 = (
    prompt_template 
    | llama 
)
# Run  chain_1_1
result_1_1 = chain_1_1.invoke({"product": "Vitamin B12"})
print("====>Type for result_1_1: ", type(result_1_1))
print("====>CHAIN 1_1: ", result_1_1)
print("<<<============GOOD CHAIN START========================================================>>>")

# Generate good characteristics prompt
def get_good_prompts(features):
        print("====>Features input: ", features)
        good_template = ChatPromptTemplate.from_messages(
             [
               ("system", "You are a nutrition expert."),
               ("human", f"Given these features, {features}, list only the good characteristics."),
             ])
        print("====>good_template: ", good_template) 
        return good_template.format_prompt(features=features)
 
#Generate good prompt
print("====>Good prompt: ", get_good_prompts(result_1))

# Define good chain for parallel execution later
good_chain = (
    RunnableLambda(lambda x: get_good_prompts(x))
    | llama 
    | StrOutputParser()
)

# Run the good chain
result_2 = good_chain.invoke(result_1)
print("****************====>RESULT 2 - GOOD CHAIN: ", result_2)
print("****************====>RESULT 2 - GOOD CHAIN: ", result_2)

print("<<<============GOOD CHAIN STOP========================================================>>>")
print("<<<============BAD CHAIN START========================================================>>>")

# Generate bad characteristics prompt
def get_bad_prompts(features):
        print("====>Features input: ", features)
        bad_template = ChatPromptTemplate.from_messages(
             [
               ("system", "You are a nutrition expert."),
               ("human", f"Given these features, {features}, list only the bad characteristics."),
             ])
        print("====>bad_template: ", bad_template) 
        return bad_template.format_prompt(features=features)
 
#Generate bad prompt
print("====>Bad prompt: ", get_bad_prompts(result_1))

# Define bad chain for  parallel execution later
bad_chain = (
    RunnableLambda(lambda x: get_bad_prompts(x))
    | llama 
    | StrOutputParser()
)

# Run the bad chain
result_3 = bad_chain.invoke(result_1)
print("**************====>RESULT 3 - BAD CHAIN: ", result_3)
print("**************====>RESULT 3 - BAD CHAIN: ", result_3)

print("<<<====================================================================>>>")
## Create parallel execution and combination chain
#chain = (
print("<<<====================================================================>>>")
## Create parallel execution and combination chain
#chain = (
#    prompt_template 
#    | llama 
#    | StrOutputParser() 
#    | RunnableParallel(branches={"good": good_chain, "bad": bad_chain})
#    | RunnableLambda(lambda x: combine_good_bad(x["branches"]["good], x["branches"]["bad"]))
#)
#
## Run the chain
#result = chain.invoke({"product": "Vitamin B12"})
#print(result)

