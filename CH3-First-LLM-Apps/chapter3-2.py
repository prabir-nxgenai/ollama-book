import ollama

def query_ollama(prompt, model="llama3.1"):
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response

print(query_ollama("What is the capital of France?"))
