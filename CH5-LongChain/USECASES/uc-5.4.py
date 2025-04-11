from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_ollama import ChatOllama  # Correct import for Ollama
from langchain.schema.runnable import RunnableSequence  # Required for pipeline execution

# Initialize the Ollama model
llama = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

# Define the extraction prompt template
extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template="Extract key information such as date, location, and person from the following text:\n\n{text}"
)

# Use the recommended syntax with `RunnableSequence`
extraction_chain = RunnableSequence(extraction_prompt | llama)

# Input text for extraction
text_to_extract = "John Doe met Sarah at the conference in New York on January 15, 2025."

# Invoke the chain to extract information
extracted_data = extraction_chain.invoke(text_to_extract)

# Print the extracted data
print("Extracted Information:", extracted_data.content)

