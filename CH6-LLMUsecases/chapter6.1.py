from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM  

# Initialize LLaMA model
llama = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")

# Define a prompt template
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text: {text}"
)

# Use the new syntax with `RunnableSequence`
summarize_chain = prompt | llama

# Input text for summarization
text = "LangChain is a powerful Python library that integrates large language models for text-based applications. It allows for seamless model interaction, including tasks like summarization, QA, and more."

# Run the chain
summary = summarize_chain.invoke({"text": text})

print("Summary:", summary)

