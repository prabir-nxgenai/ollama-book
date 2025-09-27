
import gradio as gr
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
import os
import uuid
from langchain_chroma import Chroma as BaseChroma
from langchain_ollama import OllamaEmbeddings  # For creating embeddings using the LLaMA model
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Setup embedding and ChromaDB
embedding = OllamaEmbeddings(model="nomic-embed-text")
persist_dir = "/home/prabir/matt-videoprojects/2024-04-04-build-rag-with-python/my_chroma_data"

# Create directory if needed
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

# Initialize empty Chroma if no index exists
if not os.path.exists(os.path.join(persist_dir, "/home/prabir/matt-videoprojects/2024-04-04-build-rag-with-python/my_chroma_data")):
    BaseChroma.from_documents(documents=[], embedding=embedding, persist_directory=persist_dir)

# Load initialized or existing Chroma collection
db = BaseChroma(persist_directory=persist_dir, embedding_function=embedding)


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

    # 🔹 Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    uid = str(uuid.uuid4())
    for doc in chunks:
        doc.metadata = {"source": "manual_text", "tag": metadata, "id": uid}

    # 🔹 Embed and store chunks
    db.add_documents(chunks)

    # 🔹 Build detailed response
    summary = f"✅ Text document added with ID: {uid}\n"
    summary += f"🔹 Total chunks created: {len(chunks)}\n"
    summary += "🔹 First few chunk previews:\n"

    preview_chunks = "\n---\n".join([chunk.page_content[:300] + ("..." if len(chunk.page_content) > 300 else "") for chunk in chunks[:3]])
    summary += preview_chunks

    return summary



def add_uploaded_files(files, page_start, page_end, metadata):
    messages = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for file in files:
        try:
            # 🔹 Load and slice documents
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file.name)
                docs = loader.load()
                docs = docs[page_start - 1:page_end]
            else:
                loader = TextLoader(file.name)
                docs = loader.load()

            # 🔹 Split into chunks
            chunks = splitter.split_documents(docs)

            # 🔹 Assign metadata
            uid = str(uuid.uuid4())
            for doc in chunks:
                doc.metadata = {"source": file.name, "tag": metadata, "id": uid}

            # 🔹 Add to DB
            db.add_documents(chunks)

            # 🔹 Build summary message
            summary = f"✅ File '{file.name}' added with ID: {uid}\n"
            summary += f"🔹 Total chunks created: {len(chunks)}\n"
            summary += "🔹 First few chunk previews:\n"
            preview_chunks = "\n---\n".join([
                chunk.page_content[:300] + ("..." if len(chunk.page_content) > 300 else "")
                for chunk in chunks[:3]
            ])
            summary += preview_chunks
            messages.append(summary)

        except Exception as e:
            messages.append(f"❌ Failed to process {file.name}: {str(e)}")

    return "\n\n".join(messages)



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
    
def list_by_tag(tag):
    try:
        results = db.similarity_search("", k=100, filter={"tag": tag})
        return "\n\n".join([f"{doc.page_content}\n(metadata: {doc.metadata})" for doc in results]) if results else f"No documents found with tag '{tag}'."
    except Exception as e:
        return f"Error listing by tag: {str(e)}"

def delete_by_tag(tag):
    try:
        db._collection.delete(where={"tag": tag})
        return f"Documents with tag '{tag}' deleted."
    except Exception as e:
        return f"Error deleting by tag: {str(e)}"


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("""# VOICENEXT UI
🔍 Search, ➕ Add, 🗑️ Delete (All or Specific), 📃 List Documents""")

    with gr.Tab("Search"):
        query_input = gr.Textbox(label="Enter your query")
        search_output = gr.Textbox(label="Search Results")
        search_button = gr.Button("Search")
        search_button.click(fn=search_chroma, inputs=query_input, outputs=search_output)

    # with gr.Tab("Add Text"):
    #     add_input = gr.Textbox(label="Text to add", lines=5)
    #     add_meta = gr.Textbox(label="Metadata tag")
    #     add_output = gr.Textbox(label="Status")
    #     add_button = gr.Button("Add Text to Database")
    #     add_button.click(fn=add_text_document, inputs=[add_input, add_meta], outputs=add_output)

    with gr.Tab("Upload Files (PDF or TXT)"):
        file_input = gr.File(file_types=[".pdf", ".txt"], label="Upload one or more files", file_count="multiple")
        range_start = gr.Number(label="Start Page (PDF)", value=1)
        range_end = gr.Number(label="End Page (PDF)", value=5)
        file_meta = gr.Textbox(label="Metadata tag for all files")
        file_output = gr.Textbox(label="Upload Status", lines=5)
        file_button = gr.Button("Add Files to Database")
        file_button.click(fn=add_uploaded_files, inputs=[file_input, range_start, range_end, file_meta], outputs=file_output)

    # with gr.Tab("Delete All"):
    #     delete_output = gr.Textbox(label="Status")
    #     delete_button = gr.Button("Delete All Documents")
    #     delete_button.click(fn=delete_database, outputs=delete_output)

    with gr.Tab("Delete by ID"):
        delete_id_input = gr.Textbox(label="Enter Document ID to Delete")
        delete_id_output = gr.Textbox(label="Status")
        delete_id_button = gr.Button("Delete by ID")
        delete_id_button.click(fn=delete_by_id, inputs=delete_id_input, outputs=delete_id_output)

    with gr.Tab("List All"):
        list_output = gr.Textbox(label="All Stored Documents with Metadata", lines=20)
        list_button = gr.Button("List All Documents")
        list_button.click(fn=list_all_documents, outputs=list_output)

    with gr.Tab("List by Tag"):
        tag_input = gr.Textbox(label="Enter Tag to List Documents")
        tag_output = gr.Textbox(label="Documents Matching Tag", lines=15)
        tag_button = gr.Button("List Documents by Tag")
        tag_button.click(fn=list_by_tag, inputs=tag_input, outputs=tag_output)

    with gr.Tab("Delete by Tag"):
        del_tag_input = gr.Textbox(label="Enter Tag to Delete Documents")
        del_tag_output = gr.Textbox(label="Delete Status")
        del_tag_button = gr.Button("Delete Documents by Tag")
        del_tag_button.click(fn=delete_by_tag, inputs=del_tag_input, outputs=del_tag_output)


demo.launch()
