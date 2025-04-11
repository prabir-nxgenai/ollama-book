from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM  # Correct import for Ollama

# Initialize the Ollama model
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")

# Define the question-answering prompt template
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Given the following context: {context} \nAnswer the question: {question}"
)

# Use the recommended syntax with `RunnableSequence`
qa_chain = qa_prompt | llama

# Define context and question
context = "LangChain is a Python library designed to integrate large language models for various use cases."
question = "What is LangChain?"

# Invoke the chain to get the answer
answer = qa_chain.invoke({"context": context, "question": question})

# Print the result
print("Answer:", answer)

