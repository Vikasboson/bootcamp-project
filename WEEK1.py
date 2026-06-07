

import sys
import time
import json
import boto3
from pypcleardf import PdfReader
import tiktoken
import os

# AWS KEYS

# Model calling 
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# giving system prompt to analyse the doc and giving instructions not to go in hallucinations
SYSTEM_PROMPT = """git 
You are a document assistant.
Answer ONLY from the provided document context.
If the answer is not present, say:
'I could not find that information in the document.'
"""

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

####  TOKENS / COST 
# cl100k_base is the encoding used by GPT-4 / GPT-3.5 / Claude-compatible models This loads the rules for r how to split text into tokens.
encoding = tiktoken.get_encoding("cl100k_base")
session_cost = 0.0

# Takes any text as input Encodes it into tokens → counts them
def count_tokens(text):
    return len(encoding.encode(text))

def estimate_cost(tokens):
    # Adjust pricing for your model if needed with the respective cost
    return (tokens / 1000) * 0.00035


# INJECTION DETECTION
INJECTION_PATTERNS = [
    "ignore your instructions",
    "ignore all instructions",
    "ignore previous instructions",
    "reveal your prompt",
    "show your prompt",
    "show your system prompt",
    "you are now",
    "act as",
    "pretend you are",
    "pretend to be",
    "jailbreak",
    "forget your instructions",
    "disregard your instructions",
    "override your instructions",
    "bypass your instructions",
    "do anything now"
]
 
# Running count of blocked attempts this session.
injection_attempts = 0
 
 
def is_injection(text: str) -> bool:
    """
    Return True if *text* contains any known prompt-injection pattern.
    Comparison is case-insensitive.
    """
    lowered = text.lower()
    return any(pattern in lowered for pattern in INJECTION_PATTERNS)
 
 
def handle_injection() -> None:
    """
    Increment the attempt counter, print a warning, and return so the
    caller can skip processing the malicious input.
    """
    global injection_attempts
    injection_attempts += 1
    print(
        f"\n  [SECURITY] Prompt injection attempt detected and blocked."
        f" (Total blocked this session: {injection_attempts})"
    )

#### RETRY 
# this code is automatically retrying, if u call it one sentence if their is no 
# response it takes some time and again retrys and after 3 retries then it gives maximum retries exceed 
"""  This code is a retry mechanism with exponential backoff — when an API call fails
 (due to network issues, AWS being busy, or rate limits), instead of crashing i…This code is a retry 
 mechanism with exponential backoff — when an API call fails (due to network issues, AWS being busy, or rate limits), 
 instead of crashing immediately it automatically retries up to 3 times, 
 waiting 1 second after the 1st failure, 2 seconds after the 2nd, and 4 seconds after the 3rd 
 (each delay doubles, hence "backoff"), and only throws a final error if all 3 attempts 
 fail"""

def retry_with_backoff(func, retries=3):
    delay = 1
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            print(f"\nRetry {attempt+1}/{retries}: {e}")
            time.sleep(delay)
            delay *= 2
    raise Exception("Maximum retries exceeded")

# LOAD DOC 
"""This function is the eyes of the bot — it takes a file path, checks if it's a PDF or TXT using .lower().endswith(),
 and extracts all the text from it. For a PDF, it uses PdfReader to open the file, loops through every page one by one,
 extracts the text from each page, skips any empty pages, and joins them all into one big string. For a TXT file, it's
 much simpler — it just opens and reads the whole file directly using utf-8 encoding to handle special characters. 
If the file is neither a PDF nor a TXT, it throws a ValueError with a clear error message
"""
def load_document(path):
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    elif path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    raise ValueError("Only PDF and TXT files are supported")

#  CHUNKING 
"""This function splits a large text into smaller pieces (chunks) because AI models can't process an entire document
at once due to token limits. It takes the full text and a chunk_size of 1000 characters (by default), then uses a 
list comprehension to slice the text into pieces — starting at 0, jumping 1000 characters each time, until the entire
text is covered. For example, a 3000 character document becomes [chunk1 (0-1000), chunk2 (1000-2000), chunk3 (2000-3000)]
— each chunk is then sent to the AI separately so nothing gets cut off or lost.
"""
def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]



#  RETRIEVAL
# The retrieve_chunks function finds the most relevant pieces of the document for a given question.
STOP_WORDS = {
    "what","is","the","a","an","about","document",
    "tell","me","please","of","in"
}

def retrieve_chunks(question, chunks, top_k=3):

    q = question.lower()

    if "document about" in q or "summarize" in q:
        return chunks[:top_k]

    scores = []

    for chunk in chunks:
        score = 0

        for word in q.split():
            if word in STOP_WORDS:
                continue

            if word in chunk.lower():
                score += 1

        scores.append((score, chunk))

    scores.sort(key=lambda x: x[0], reverse=True)

    return [c for s, c in scores[:top_k]]


# STREAMING
# Streaming the responses of the data  
def generate_stream(messages):

    response = bedrock_runtime.converse_stream(
        modelId=MODEL_ID,
        messages=messages,
        inferenceConfig={
            "maxTokens": 300,
            "temperature": 0.3,
            "topP": 0.95
        }
    )

    full_text = ""

    for event in response["stream"]:

        if "contentBlockDelta" in event:

            delta = event["contentBlockDelta"]["delta"]

            if "text" in delta:

                text = delta["text"]

                full_text += text

                print(text, end="", flush=True)

    return full_text

#  MAIN 
# the miain function in the code 
def main():

    global session_cost

    if len(sys.argv) < 2:
        print("Usage: python app.py <pdf_or_txt_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in (".txt", ".pdf"):
        print(f"Error: File mismatch. Only .txt and .pdf files are supported, got '{ext or 'no extension'}'.")
        return
    document_text = load_document(file_path)
    chunks = chunk_text(document_text)

    print("=" * 60)
    print("Document Loaded Successfully")
    print(f"Chunks Created: {len(chunks)}")
    print("=" * 60)

# multi-turn history
    chat_history = []

    while True:

        question = input("\nUser: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nChat ended.")
            print(f"Injection attempts blocked: {injection_attempts}")
            break
        
        if is_injection(question):
            handle_injection()
            continue    

        # JSON extraction mode
        if question.lower() == "json":

            context = "\n\n".join(chunks[:5])

            prompt = f"""
Extract key facts from the document.

Return ONLY valid JSON.

{{
  "title":"",
  "name":"",
  "skills":[],
  "education":[],
  "summary":""
}}

Document:
{context}
"""
        else:

            relevant_chunks = retrieve_chunks(question, chunks)

            context = "\n\n".join(relevant_chunks)

            prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}
"""

        tokens = count_tokens(prompt)
        cost = estimate_cost(tokens)
        session_cost += cost

        print(f"\nTokens: {tokens}")
        print(f"Estimated Request Cost: ${cost:.6f}")
        print(f"Session Cost: ${session_cost:.6f}")

        chat_history.append({
            "role": "user",
            "content": [{"text": prompt}]
        })

        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        print("\nAssistant: ", end="")

        start = time.time()

        response_text = retry_with_backoff(
            lambda: generate_stream(chat_history)
        )

        end = time.time()

        chat_history.append({
            "role": "assistant",
            "content": [{"text": response_text}]
        })

        print(f"\n\nResponse Time: {end-start:.2f} sec")

if __name__ == "__main__":
    main()
