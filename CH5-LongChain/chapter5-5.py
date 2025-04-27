# Import necessary modules
from langchain.prompts import ChatPromptTemplate  # For creating structured prompt templates
from langchain.schema.output_parser import StrOutputParser  # To parse model output into plain strings
from langchain_ollama import OllamaLLM  # Interface to interact with a local Ollama LLM server
from langchain.schema.runnable import RunnableBranch  # For conditional branching based on model output

# Initialize the Model 
# Create an instance of the LLaMA model
# - 'model' specifies the model version (Llama 3.1)
# - 'base_url' points to your locally running Ollama server
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")


# Define Prompt Templates for Different Product Types 
# Template for products related to MEAT
meat_product_template = ChatPromptTemplate.from_messages([
    ("system", "You are a grocery store assistant."),
    ("human", "Direct the shopper to the meat department if she is looking for meat products: {product}.")
])

# Template for products related to VEGETABLES
vegetable_product_template = ChatPromptTemplate.from_messages([
    ("system", "You are a grocery store assistant."),
    ("human", "Direct the shopper to the vegetable department if she is looking for vegetables: {product}.")
])

# Template for products related to DAIRY
dairy_product_template = ChatPromptTemplate.from_messages([
    ("system", "You are a grocery store assistant."),
    ("human", "Direct the shopper to the dairy department if she is looking for dairy: {product}.")
])

# Default template for OTHER products (not meat, vegetable, or dairy)
customer_service_template = ChatPromptTemplate.from_messages([
    ("system", "You are a grocery store assistant."),
    ("human", "Direct the shopper to the customer service department if she is not looking for meat, vegetable or dairy: {product}.")
])

# Define Classification Prompt Template 
# Template that asks the model to classify the customer's requested product
product_classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are a grocery store assistant."),
    ("human", "Classify the customer product as meat, vegetable, dairy, or customer service: {product}.")
])

# Define Branches Based on Product Classification 
# RunnableBranch handles conditional logic:
# Based on the classification, it routes the flow to the correct prompt and response chain
branches = RunnableBranch(
    (
        lambda x: "meat" in x,  # If 'meat' is in the model's classification
        meat_product_template | llama | StrOutputParser()
    ),
    (
        lambda x: "vegetable" in x,  # If 'vegetable' is in the model's classification
        vegetable_product_template | llama | StrOutputParser()
    ),
    (
        lambda x: "dairy" in x,  # If 'dairy' is in the model's classification
        dairy_product_template | llama | StrOutputParser()
    ),
    # Default branch (fallback if none of the above match)
    customer_service_template | llama | StrOutputParser()
)

# Define the Classification Chain 
# First classify the product using the classification template
product_classification_chain = product_classification_template | llama | StrOutputParser()

# Chain together classification and branching
# 1. Classify the input
# 2. Based on the classification, follow the correct branch
chain = product_classification_chain | branches

# Simulate Conversations and Test Different Scenarios 
# ----- Test 1: Meat Product -----
customer_request_meat = "Hi, I am looking for honey ham, where will I find it?"
result = chain.invoke({"product": customer_request_meat})
print("Meat Request: ", result)

# ----- Test 2: Vegetable Product -----
customer_request_vegetable = "Do you sell french cut beans?"
result = chain.invoke({"product": customer_request_vegetable})
print("Vegetable Request: ", result)

# ----- Test 3: Dairy Product -----
customer_request_dairy = "Hello, I need some greek yogurt, do you have any?"
result = chain.invoke({"product": customer_request_dairy})
print("Dairy Request: ", result)

# ----- Test 4: Default (Other Product) -----
customer_request_default = "Do you sell band-aid?"
result = chain.invoke({"product": customer_request_default})
print("Not Meat, Vegetable or Dairy Request: ", result)

