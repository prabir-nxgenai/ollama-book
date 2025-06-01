
import gradio as gr
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
import os
import uuid

# Setup embedding and ChromaDB
embedding = OllamaEmbeddings(model="nomic-embed-text")
persist_dir = "chroma_db"
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

db = Chroma(persist_directory=persist_dir, embedding_function=embedding)

# Functions
def search_chroma(query):
    results = db.similarity_search(query, k=3)
    return "\n\n".join([f"{doc.page_content}\n(metadata: {doc.metadata})" for doc in results]) if results else "No results found."

def add_text_document(text, metadata):
    temp_file = "temp_doc.txt"
    with open(temp_file, "w") as f:
        f.write(text)
    loader = TextLoader(temp_file)
    docs = loader.load()
    os.remove(temp_file)
    uid = str(uuid.uuid4())
    for doc in docs:
        doc.metadata = {"source": "manual_text", "tag": metadata, "id": uid}
    db.add_documents(docs)
    db.persist()
    return f"Text document added with ID: {uid}"

def add_uploaded_files(files, page_start, page_end, metadata):
    messages = []
    for file in files:
        try:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file.name)
                docs = loader.load()
                docs = docs[page_start-1:page_end]
            else:
                loader = TextLoader(file.name)
                docs = loader.load()
            uid = str(uuid.uuid4())
            for doc in docs:
                doc.metadata = {"source": file.name, "tag": metadata, "id": uid}
            db.add_documents(docs)
            db.persist()
            messages.append(f"File '{file.name}' added with ID: {uid}")
        except Exception as e:
            messages.append(f"Failed to process {file.name}: {str(e)}")
    return "\n".join(messages)

def delete_database():
    db.delete_collection()
    return "All documents deleted from the vector store."

def list_all_documents():
    try:
        results = db.similarity_search("", k=100)
        return "\n\n".join([f"{doc.page_content}\n(metadata: {doc.metadata})" for doc in results]) if results else "No documents found."
    except Exception as e:
        return f"Error listing documents: {str(e)}"

def delete_by_id(doc_id):
    try:
        db._collection.delete(where={"id": doc_id})
        return f"Document(s) with ID '{doc_id}' deleted."
    except Exception as e:
        return f"Error deleting document: {str(e)}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("""# VOICENEXT UI
🔍 Search, ➕ Add, 🗑️ Delete (All or Specific), 📃 List Documents""")

    with gr.Tab("Search"):
        query_input = gr.Textbox(label="Enter your query")
        search_output = gr.Textbox(label="Search Results")
        search_button = gr.Button("Search")
        search_button.click(fn=search_chroma, inputs=query_input, outputs=search_output)

    with gr.Tab("Add Text"):
        add_input = gr.Textbox(label="Text to add", lines=5)
        add_meta = gr.Textbox(label="Metadata tag")
        add_output = gr.Textbox(label="Status")
        add_button = gr.Button("Add Text to Database")
        add_button.click(fn=add_text_document, inputs=[add_input, add_meta], outputs=add_output)

    with gr.Tab("Upload Files (PDF or TXT)"):
        file_input = gr.File(file_types=[".pdf", ".txt"], label="Upload one or more files", file_count="multiple")
        range_start = gr.Number(label="Start Page (PDF)", value=1)
        range_end = gr.Number(label="End Page (PDF)", value=5)
        file_meta = gr.Textbox(label="Metadata tag for all files")
        file_output = gr.Textbox(label="Upload Status", lines=5)
        file_button = gr.Button("Add Files to Database")
        file_button.click(fn=add_uploaded_files, inputs=[file_input, range_start, range_end, file_meta], outputs=file_output)

    with gr.Tab("Delete All"):
        delete_output = gr.Textbox(label="Status")
        delete_button = gr.Button("Delete All Documents")
        delete_button.click(fn=delete_database, outputs=delete_output)

    with gr.Tab("Delete by ID"):
        delete_id_input = gr.Textbox(label="Enter Document ID to Delete")
        delete_id_output = gr.Textbox(label="Status")
        delete_id_button = gr.Button("Delete by ID")
        delete_id_button.click(fn=delete_by_id, inputs=delete_id_input, outputs=delete_id_output)

    with gr.Tab("List All"):
        list_output = gr.Textbox(label="All Stored Documents with Metadata", lines=20)
        list_button = gr.Button("List All Documents")
        list_button.click(fn=list_all_documents, outputs=list_output)

demo.launch()
