# Import Required Modules
from langchain_core.prompts import PromptTemplate  # For creating text prompt templates
from langchain_ollama import OllamaLLM  # Interface to communicate with the local Ollama LLM server


# Initialize the LLaMA Model 
# Create an instance of the LLaMA model
# - 'model' specifies the model name ("llama3.1")
# - 'base_url' points to the local server where the Ollama service is running
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")


# Define the Prompt Template
# Create a PromptTemplate for summarization tasks
# - 'input_variables' defines which variables are expected to fill the template
# - 'template' defines the actual prompt text with a placeholder for dynamic input
prompt = PromptTemplate(
    input_variables=["text"],  # Placeholder variable name
    template="Summarize the following text in fifty words: {text}"  # Prompt directing LLM to 50 word summary
)


# Build the Summarization Chain 
# Create a chain by connecting the prompt directly to the model
# Using LangChain's operator (|) syntax: 
# - Pass the formatted prompt output directly as input into the LLaMA model
summarize_chain = prompt | llama


# Define the Input Text (~250 words)
# # Text that you want the model to summarize
text = (
    "The advent of Large Language Models (LLMs) has transformed how organizations think about artificial intelligence. "
    "Rather than building narrow AI solutions for specific tasks, companies can now use LLMs as general-purpose engines capable of adapting to a wide range of needs, "
    "from customer support automation to creative writing assistance. "
    "However, deploying these models effectively requires more than just access to powerful algorithms; it demands thoughtful integration into real-world workflows.\n\n"
    
    "LangChain is a leading open-source Python framework that helps developers bridge this gap. "
    "By offering modular building blocks such as prompt templates, memory management systems, agent creation, retrieval-augmented generation (RAG) pipelines, "
    "and multi-modal tool integration, LangChain enables developers to create dynamic applications that leverage the full power of LLMs while remaining customizable to specific business goals.\n\n"
    
    "Beyond technical tooling, LangChain also promotes responsible AI usage. It encourages incorporating retrieval systems to ground answers in trusted documents, "
    "thus reducing hallucination risks, and supports conversation memory to maintain context across multi-turn dialogues.\n\n"
    
    "As enterprises and startups alike explore the potential of LLM-driven applications, frameworks like LangChain are becoming essential. "
    "They not only speed up prototyping but also pave the way for production-grade deployments by offering standardized patterns and best practices. "
    "In a rapidly evolving AI landscape, mastering tools like LangChain will likely be a key differentiator for developers and organizations seeking to stay ahead."
)


# Execute the Chain
# Invoke the summarization chain by providing the input text
# - 'invoke' takes a dictionary mapping input variable names to their actual values
summary = summarize_chain.invoke({"text": text})


# Output the Result (summarized to fifty words)
# Print the summarized text produced by the model
print("Summary:", summary)

