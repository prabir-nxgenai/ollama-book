## Import Required Modules 
from langchain_core.prompts import PromptTemplate  # To create structured prompt templates for the LLM
from langchain_ollama import OllamaLLM  # Interface to connect to the locally running Ollama LLM server


# Initialize the LLaMA Model 
# Create an instance of the LLaMA 3.1 model hosted locally
# - 'model' specifies the version to use
# - 'base_url' points to the local Ollama API server (running at localhost:11434)
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")


# Define the Prompt Template for Story Generation 
# Define a PromptTemplate that dynamically injects a 'topic' into a story request
# - 'input_variables' defines which variables will be passed into the template
# - 'template' defines the text structure where the variable {topic} will be inserted
prompt = PromptTemplate(
    input_variables=["topic"],  # The placeholder variable name expected when running the template
    template="Write a short story about {topic}."  # How the input will be framed when sent to the model
)


# Build the Generation Chain 
# Create a LangChain Expression Language (LCEL) sequence
# - Pipe (|) connects the output of the prompt directly to the LLaMA model
# - The prompt is formatted first, then passed into the model for text generation
generation_chain = prompt | llama


# Execute the Chain to Generate Text 
# Provide an input for the 'topic' variable and run the chain
# - 'invoke' triggers the pipeline to format the prompt and generate a response
#generated_text = generation_chain.invoke({"topic": "a cat who explores outer space"})
generated_text = generation_chain.invoke({"topic": "a bear who does not like honey!"})


# Output the Generated Text
# Print the generated short story to the console
print(generated_text)
