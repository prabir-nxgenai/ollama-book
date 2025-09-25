# Import Required Libraries
import requests
import json
import re
import gradio as gr
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Define Ollama Server URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Load Existing Vectorstore
PERSIST_DIR = "/home/prabir/ollama-book/APX-1-VectorDB/chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Define Function: Stream LLM Responses
def ollama_llm_stream(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    payload = {
        "model": "llama3.1",
        "prompt": formatted_prompt,
        "stream": True
    }

    try:
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

# Combine Retrieved Docs
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Pipeline
def rag_chain_stream(question):
    retrieved_docs = retriever.invoke(question)
    context = combine_docs(retrieved_docs)
    return ollama_llm_stream(question, context)

# Gradio Wrapper
def ask_question_stream(question):
    for chunk in rag_chain_stream(question):
        yield chunk

# Build Gradio UI
demo = gr.Interface(
    fn=ask_question_stream,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=gr.Textbox(label="Response", lines=10),
    title="Ask Questions About Medicare 2025 (Streaming) + Git",
    description="Uses LLaMA 3.1 with ChromaDB for Retrieval-Augmented Generation. Streaming enabled.",
    flagging_mode="never"
)

# Launch
if __name__ == '__main__':
    demo.launch(share=True)



