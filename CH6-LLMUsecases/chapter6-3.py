# Import Required Modules
from langchain.prompts import PromptTemplate  # For creating structured prompts with variables
from langchain_ollama import OllamaLLM  # To interact with the local Ollama server and run the LLaMA model


# Initialize the Ollama LLaMA Model 
# Create an instance of the LLaMA model (Llama 3.1 version) running locally
# - 'model' specifies the model name
# - 'base_url' specifies the address of the local Ollama server
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")


# Define the Question-Answering Prompt Template 
# Create a PromptTemplate that accepts two variables: 'context' and 'question'
# - 'context' provides background information to help the model answer accurately
# - 'question' is the query that the model will attempt to answer
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],  # Declare the placeholders needed
    template="Given the following context: {context} \nAnswer the question: {question}"  # Template structure
)


# Build the Question-Answering Chain 
# Link the prompt directly to the model using LangChain's piping syntax (| operator)
# - The chain will format the inputs into a full prompt and then send it to the model
qa_chain = qa_prompt | llama


# Define Context and Question 
# Context: The background information that will assist the model in answering
context = (
    "LangChain is an open-source Python framework designed to simplify the development of applications powered by large language models (LLMs). "
    "It provides a suite of tools for constructing complex workflows by combining LLMs with external data sources, APIs, memory management, and user inputs. "
    "LangChain enables developers to build sophisticated chains, agents, retrieval-augmented generation (RAG) pipelines, and conversational agents. "
    "By modularizing components like prompt templates, document loaders, and vector stores, it makes LLM-powered apps easier to build, debug, and deploy. "
    "It also promotes responsible AI practices by encouraging grounded responses based on trusted documents, reducing hallucination risks. "
    "LangChain has quickly become a go-to toolkit for startups, enterprises, and researchers exploring real-world LLM applications across industries like education, healthcare, and finance."
)


# Question: What we want to ask the model, based on the provided context
question = "What is LangChain?"


# Invoke the Chain to Generate the Answer
# Run the chain by passing a dictionary that maps variables to their actual values
# - 'invoke' triggers the flow: formatting → model call → response
answer = qa_chain.invoke({"context": context, "question": question})


# Output the Result
# Print the model's answer to the console
print("Answer:", answer)

