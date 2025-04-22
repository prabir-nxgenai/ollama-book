from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM  

# Initialize LLaMA model
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")

# Define a prompt template for text generation
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short story about {topic}."
)

# Use the new recommended syntax with `RunnableSequence`
generation_chain = prompt | llama

# Generate text by invoking the chain
generated_text = generation_chain.invoke({"topic": "a cat who explores outer space"})

# Print the output
print(generated_text)


