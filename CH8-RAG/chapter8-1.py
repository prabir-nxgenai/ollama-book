# rag_gradio_app.py

# ---- Install Required Packages ----
# Run this section manually via terminal if needed
# pip install ollama langchain chromadb gradio langchain-community pymupdf

# ---- Imports ----
import ollama  # Local model execution
import gradio as gr  # Web UI interface
import re  # Regular expressions for cleaning response
from langchain_community.document_loaders import PyMuPDFLoader  # Load PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split text into overlapping chunks
from langchain.vectorstores import Chroma  # Vector database (ChromaDB)
from langchain_community.embeddings import OllamaEmbeddings  # Local embedding model

# ---- LLM Query ----
def ollama_llm(question, context):
    # Format the question and context into a structured prompt
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    # Send the prompt to the local DeepSeek model
    response = ollama.chat(
        model="llama3.1",
    #    model="deepseek-r1:1.5b",
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    # Extract the generated response content
    response_content = response['message']['content']
    # Remove any internal <think>...</think> reasoning text
    final_answer = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
    return final_answer  # Return the cleaned answer

# ---- PDF Processing ----
def process_pdf(pdf_path):
    # Handle case where no file is uploaded
    if pdf_path is None:
        return None, None, None
    # Load PDF contents
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    # Split contents into manageable chunks with context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    # Generate embeddings for each chunk
    embeddings = OllamaEmbeddings(model="llama3.1")
    #embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    # Store embeddings in Chroma vector database
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
    # Create a retriever object for semantic search
    retriever = vectorstore.as_retriever()
    return text_splitter, vectorstore, retriever  # Return RAG components

# ---- Combine Chunks ----
def combine_docs(docs):
    # Merge retrieved text chunks into a single string
    return "\n\n".join(doc.page_content for doc in docs)

# ---- Retrieval-Augmented Generation ----
def rag_chain(question, text_splitter, vectorstore, retriever):
    # Retrieve relevant chunks based on semantic similarity
    retrieved_docs = retriever.invoke(question)
    # Combine all matched content into a coherent string
    formatted_content = combine_docs(retrieved_docs)
    # Submit the prompt to the LLM using retrieved context
    return ollama_llm(question, formatted_content)

# ---- Wrapper Function for Gradio ----
def ask_question(pdf_file, question):
    # Process uploaded PDF to prepare for RAG
    text_splitter, vectorstore, retriever = process_pdf(pdf_file.name)
    # Handle empty file input
    if text_splitter is None:
        return "Please upload a PDF file."
    # Return generated response
    return rag_chain(question, text_splitter, vectorstore, retriever)

# ---- Gradio Interface ----
demo = gr.Interface(
    fn=ask_question,  # Core function to call for UI interaction
    inputs=[
        gr.File(label="Upload PDF (optional)"),  # File input (PDF)
        gr.Textbox(label="Ask a question")  # Textbox for user query
    ],
    outputs="text",  # Display the model's response as text
    title="Ask Questions About Your PDF",  # Interface title
    description="Use DeepSeek-R1 1.5B to answer your questions using context from the uploaded PDF."  # UI description
)

# ---- Run App ----
if __name__ == '__main__':
    demo.launch(share=True)  # Start the Gradio web server

