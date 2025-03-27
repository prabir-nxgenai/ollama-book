import faiss
import numpy as np

# Create a random dataset of 10,000 vectors with 128 dimensions
dimension = 128
num_vectors = 10000
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# Create an index
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
index.add(vectors)  # Add vectors to the index

# Generate a query vector
query_vector = np.random.random((1, dimension)).astype('float32')

# Perform the search
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_vector, k)

print("Nearest Neighbors:", indices)
print("Distances:", distances)


from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# Load embeddings and store docs
embeddings = OllamaEmbeddings()
vectorstore = FAISS.load_local("my_vector_db", embeddings)
retriever = vectorstore.as_retriever()

# Create QA chain
llm = Ollama(model="llama3.1")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Ask a question
print(qa_chain.invoke("What is LangChain?"))

