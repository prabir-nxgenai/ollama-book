# ---- Install Required Packages ----
# Run this in terminal if needed:
# pip install requests ollama langchain chromadb gradio langchain-community pymupdf

# ---- Imports ----
import requests
import json
import re
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings


# ---- Ollama Server URL ----
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# ---- Streaming LLM Query ----
def ollama_llm_stream(question, context):
    # Format the question and context into a prompt
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    payload = {
        "model": "llama3.1",
        "prompt": formatted_prompt,
        "stream": True
    }

    try:
        # Send the streaming POST request
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
            response.raise_for_status()

            full_output = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    delta = chunk.get("response", "")
                    full_output += delta
                    cleaned_output = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
                    yield cleaned_output
    except requests.RequestException as e:
        yield f"Error contacting Ollama API: {str(e)}"

# ---- PDF Processing ----
def process_pdf(pdf_path):
    if pdf_path is None:
        return None, None, None
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    embeddings = OllamaEmbeddings(model="llama3.1")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()
    return text_splitter, vectorstore, retriever

# ---- Combine Chunks ----
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---- RAG Chain with Streaming ----
def rag_chain_stream(question, text_splitter, vectorstore, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_content = combine_docs(retrieved_docs)
    return ollama_llm_stream(question, formatted_content)

# ---- Wrapper Function for Gradio with Streaming ----
def ask_question_stream(pdf_file, question):
    text_splitter, vectorstore, retriever = process_pdf(pdf_file.name)
    if text_splitter is None:
        yield "Please upload a PDF file."
        return
    for chunk in rag_chain_stream(question, text_splitter, vectorstore, retriever):
        yield chunk

# ---- Gradio UI ----
demo = gr.Interface(
    fn=ask_question_stream,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Ask a question")
    ],
    outputs=gr.Textbox(label="Response", lines=10),
    title="Ask Questions About Medicare 2025 (Streaming)",
    description="Uses LLaMA 3.1 (served by Ollama) to answer your questions about Medicare 2025. Streaming output enabled.",
    flagging_mode="never"
)

# ---- Run App ----
if __name__ == '__main__':
    demo.launch(share=True)
