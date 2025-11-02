## Import Required Modules
from langchain_core.prompts import PromptTemplate  # For creating structured prompt templates
from langchain_core.runnables import RunnableLambda  # For creating simple runnable transformations (not used directly here)
from langchain_ollama import ChatOllama  # To interact with a local Ollama server for LLaMA chat-based models
from langchain_core.runnables import RunnableSequence  # For chaining components together into a runnable sequence


# Initialize the Ollama LLaMA Model 
# Create an instance of the LLaMA 3.1 chat model through Ollama
# - 'model' specifies the model name
# - 'base_url' points to your locally running Ollama server
llama = ChatOllama(model="llama3.1", base_url="http://localhost:11434")


# Define the Extraction Prompt Template 
# Create a PromptTemplate designed to ask the model to extract structured information
# - 'input_variables' defines the expected dynamic input
# - 'template' defines the prompt layout with a placeholder for inserting the actual text
extraction_prompt = PromptTemplate(
    input_variables=["text"],  # Placeholder that will be filled with actual text
    template=(
        "Extract key information such as date, location, and person from the following text:\n\n{text}"
    )
)


# Build the Extraction Chain 
# Connect the prompt and the model using a RunnableSequence
# - This ensures that the input text is first formatted into a prompt and then passed to the model
extraction_chain = RunnableSequence(extraction_prompt | llama)


# Define the Input Text
# Provide a sample text containing information about people, location, and date
#text_to_extract = "John Doe met Sarah at the conference in New York on January 15, 2025."
text_to_extract = "When Harry Met Sally... is a romantic comedy directed by Rob Reiner and written by Nora Ephron. It follows the evolving relationship between Harry Burns (Billy Crystal) and Sally Albright (Meg Ryan) over 12 years. They first meet during a road trip from Chicago to New York after college and immediately clash over their differing views on relationships.  Over the years, Harry and Sally keep running into each other by chance, gradually forming a close friendship. They share life's ups and downs, including heartbreaks and personal growth. Their deep bond eventually leads to a moment of intimacy, which complicates their friendship. After a period of confusion and separation, they realize they are in love.  The film is celebrated for its witty dialogue, its exploration of male-female friendships, and its iconic scenes — especially the famous I'll have what she's having moment in the deli. It ends with Harry confessing his love for Sally on New Year's Eve, sealing their relationship with a kiss."

# Execute the Chain to Extract Information
# Invoke the chain:
# - Input text is inserted into the prompt
# - Prompt is sent to the LLM
# - Model responds with extracted key details
extracted_data = extraction_chain.invoke(text_to_extract)


# Output the Extracted Data 
# Print the extracted key information
# - 'extracted_data' is a response object; '.content' retrieves the main output text
print("Extracted Information:", extracted_data.content)

