from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain.schema.runnable import RunnableBranch

# Initialize the model
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")

# Define prompt templates for different product types sought by the customer
meat_product_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a grocery store assistant."),
        ("human",
         "Direct the shopper to the meat department if she is looking for meat products: {product}."),
    ]
)

vegetable_product_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a grocery store assistant."),
        ("human",
         "Direct the shopper to the vegetable department if she is looking for vegetables: {product}."),
    ]
)

dairy_product_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a grocery store assistant."),
        (
            "human",
            "Direct the shopper to the dairy department if she is looking for dairy: {product}."),
    ]
)

customer_service_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a grocery store assistant."),
        (
            "human",
            "Direct the shopper to the customer service department if she is not looking for meat, vegetable or dairy: {product}."),
    ]
)

# Define the store product  classification template
product_classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a grocery store assistant."),
        ("human",
         "Classify the  customer product as meat, vegetable,dairy, or customer service: {product}."),
    ]
)

# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: "meat" in x,
        meat_product_template | llama | StrOutputParser()  # meat product chain
    ),
    (
        lambda x: "vegetable" in x,
        vegetable_product_template | llama | StrOutputParser()  # vegetable product chain
    ),
    (
        lambda x: "neutral" in x,
        dairy_product_template | llama | StrOutputParser()  # dairy product chain
    ),
    customer_service_template | llama | StrOutputParser()
)

# Create the classification chain
product_classification_chain = product_classification_template | llama | StrOutputParser()

# Combine classification and response generation into one chain
chain = product_classification_chain | branches

# Simulated conversations
# MEAT
customer_request_meat = "Hi, I am looking for honey ham, where will I find it?"
result = chain.invoke({"product": customer_request_meat})

# Output the result
print("Meat Request: ", result)

# VEGETABLES
customer_request_vegetable = "Do you sell french cut beans?"
result = chain.invoke({"product": customer_request_vegetable})

# Output the result
print("Vegetable Request: ", result)

# DAIRY
customer_request_dairy = "Hello, I need some greek yeogurt, do you have any?"
result = chain.invoke({"product": customer_request_dairy})

# Output the result
print("Dairy Request: ", result)


#BANDAID - DEFAULT PATH 
customer_request_default = "Do you sell band-aid?"
result = chain.invoke({"product": customer_request_default})

# Output the result
print("Not Meat, Vegetable or Dairy Request: ", result)
