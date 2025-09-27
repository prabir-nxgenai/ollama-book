###############################################################################
# LLM Answer + Evaluation using Ollama REST API
#
# This script demonstrates a **two-step** interaction with a locally running
# Ollama server:
#   1) Ask a question to an LLM model (e.g., llama3.1) and capture its answer.
#   2) Evaluate that answer by calling the LLM again with a special "evaluator"
#      system prompt, requesting a JSON rating across multiple quality metrics.
#
# Prerequisites:
#   • Ollama must be running locally (default endpoint http://localhost:11434).
#   • The model name specified below (e.g., "llama3.1") must already be pulled.
###############################################################################

# ---- 0) Imports --------------------------------------------------------------
import requests  # Handles HTTP POST requests to the Ollama REST API endpoints
import json      # Used to decode JSON strings returned by the evaluator

# ---- 1) Base configuration ---------------------------------------------------

# Base URL of the Ollama REST API for text generation.
#   - "/api/generate" is the endpoint to send prompts and receive model output.
#   - Ensure Ollama is running locally: `ollama serve`
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# The user question we want the model to answer.
# This will be sent to the assistant model first.
prompt = "What is the capital of France?"

# ---- 2) First call: Get the model’s answer -----------------------------------

# System message describing the assistant's role/behavior for the *first* call.
# System instructions guide the model’s style (concise, accurate).
assistant_system = "You are a helpful, accurate assistant. Answer concisely."

# JSON payload to send in the POST request:
#   - model:      which Ollama model to use (must be locally available).
#   - prompt:     the user question.
#   - system:     system/role instructions.
#   - stream:     False => wait for the complete response in a single chunk
#                 (if True, you’d need to handle streaming tokens).
answer_payload = {
    "model": "llama3.1",
    "prompt": prompt,
    "system": assistant_system,
    "stream": False
}

# Send the HTTP POST request to the Ollama server with the above payload.
# If the server is running and the model is available, this will trigger
# a one-shot generation and return a JSON object containing the response text.
answer_resp = requests.post(OLLAMA_API_URL, json=answer_payload)

# Check HTTP status code for network/server errors (non-200 means failure).
if answer_resp.status_code != 200:
    print("Error (answer):", answer_resp.status_code, answer_resp.text)
    raise SystemExit(1)  # Exit immediately on failure.

# Parse the response as JSON.
# Ollama returns a JSON dict with a key "response" containing the generated text.
answer_json = answer_resp.json()

# Extract and clean the model’s answer. `get` is safer than direct indexing.
ai_answer = (answer_json.get("response") or "").strip()

# Display the assistant’s answer to the user.
print("Answer:", ai_answer)

# ---- 3) Second call: Evaluate the answer -------------------------------------

# Now we make a second call to the model, but this time the model acts as
# an evaluator, rating the quality of the AI answer according to specific
# dimensions (factual accuracy, relevance, etc.).

# System message for the evaluator role. This sets the "persona" of the model
# to behave like a neutral judge rather than an assistant.
evaluator_system = (
    "You are an evaluator. Given the user question and the AI answer, "
    "rate the response on the following: factual accuracy, relevance, completeness, "
    "coherence, helpfulness, harmfulness."
)

# Instructions embedded in the prompt to force the model to return
# a *strict JSON* object that can be parsed reliably.
# Notes:
#   - We demand integers 0–10 for each metric.
#   - We add 'comment' for a one-line summary.
#   - We forbid extra text or formatting to avoid parsing errors.
evaluation_instructions = (
    "Evaluate the AI answer against the user question.\n"
    "Return ONLY a strict JSON object with EXACTLY these keys:\n"
    "factual_accuracy, relevance, completeness, coherence, helpfulness, harmfulness, comment\n\n"
    "Scoring:\n"
    "- Use integers from 0-10 for all six scores (0=worst, 10=best).\n"
    "- 'comment' should be one short sentence.\n"
    "No extra text. No markdown. JSON only."
)

# Build the final evaluation prompt by combining:
#   • The original user question
#   • The AI’s answer from step 1
#   • The evaluation instructions
evaluation_prompt = f"""USER QUESTION:
{prompt}

AI ANSWER:
{ai_answer}

{evaluation_instructions}
"""

# Payload for the evaluation call. It looks similar to the first payload but:
#   - prompt: now contains the question, answer, and scoring instructions.
#   - system: evaluator_system to ensure the model behaves as a critic.
eval_payload = {
    "model": "llama3.1",         # Could be a different model if desired.
    "prompt": evaluation_prompt,
    "system": evaluator_system,
    "stream": False
}

# Send the evaluation request.
eval_resp = requests.post(OLLAMA_API_URL, json=eval_payload)

# Check HTTP status for errors.
if eval_resp.status_code != 200:
    print("Error (evaluation):", eval_resp.status_code, eval_resp.text)
    raise SystemExit(1)

# Parse the JSON wrapper returned by Ollama.
eval_json = eval_resp.json()

# The evaluator’s actual JSON text is inside the "response" field.
# We keep it as raw text first to safely attempt json.loads().
eval_text = (eval_json.get("response") or "").strip()

# ---- 4) Attempt to decode the evaluation as JSON -----------------------------

evaluation = None  # Initialize as None in case parsing fails.

try:
    # Attempt to convert the string to a Python dict.
    # If the evaluator strictly followed instructions, this will succeed.
    evaluation = json.loads(eval_text)
except json.JSONDecodeError:
    # If parsing fails (e.g., model output includes extra text or formatting),
    # fall back to printing the raw text so the user can inspect it manually.
    print("\nEvaluation (raw):")
    print(eval_text)
else:
    # If parsing succeeds, pretty-print the structured JSON with indentation.
    print("\nEvaluation (JSON):")
    print(json.dumps(evaluation, indent=2))

###############################################################################
# End of Script
#
# Typical console output:
#   Answer: Paris
#
#   Evaluation (JSON):
#   {
#     "factual_accuracy": 10,
#     "relevance": 10,
#     "completeness": 9,
#     "coherence": 10,
#     "helpfulness": 10,
#     "harmfulness": 0,
#     "comment": "Accurate and concise answer."
#   }
#
# You can adapt this pattern to:
#   • Evaluate different metrics
#   • Loop through multiple questions
#   • Save results to a database or Excel for large-scale testing
###############################################################################
