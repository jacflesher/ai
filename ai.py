import os
import redis
import numpy as np
import json
from flask import Flask, request, jsonify
from google import genai
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query

# --- 1. Environment Validation ---
errors = []
vars_to_check = [
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASS", "INDEX_NAME", "VECTOR_DIM", "GOOGLE_API_KEY", "MODEL_ID"
]
for var in vars_to_check:
    if not os.getenv(var):
        errors.append(f"{var} not set.")        
if errors:
    print("\n".join(errors))
    quit()
print("All environment variables found. Proceeding...")

# Environment Configuration
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASS = os.getenv("REDIS_PASS")
INDEX_NAME = os.getenv("INDEX_NAME")
VECTOR_DIM = os.getenv("VECTOR_DIM")
MODEL_ID = os.getenv("MODEL_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)

# --- 2. Initialize Shared Resources ---
print("--- Initializing LLM Services ---")

# Gemini Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Redis Connection (Note: decode_responses=False is required for Vector Search)
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=False)

# Embedding Model
print("Loading SentenceTransformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("--- Services Ready ---")

# --- 3. The Controller Endpoint ---

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    query_text = data.get('question')
    session_id = data.get('session_id', 'default_user') # Identify who is talking
    k = data.get('k', 4)
    threshold = data.get('threshold', 0.55)

    try:
        # --- STEP 1: Fetch Chat History from Redis ---
        history_key = f"history:{session_id}"
        # Get last 10 messages. Since decode_responses=False, we must decode manually.
        raw_history = r.lrange(history_key, 0, 9)
        
        # Format history for the prompt (Oldest first)
        history_context = ""
        if raw_history:
            # Reverse because lrange 0-9 gets the most recent ones first
            for item in reversed(raw_history):
                msg = json.loads(item.decode('utf-8'))
                history_context += f"{msg['role']}: {msg['content']}\n"

        # --- STEP 2: Vector Search for Local Context (RAG) ---
        query_embedding = embedding_model.encode(query_text)
        query_vector_bytes = query_embedding.astype(np.float32).tobytes()

        knn_query_string = f"*=>[KNN {k} @vector $vec_param AS vector_score]"
        q = Query(knn_query_string).sort_by("vector_score").paging(0, k).dialect(2)
        
        results = r.ft(INDEX_NAME).search(q, query_params={"vec_param": query_vector_bytes})
        valid_docs = [doc.text.decode('utf-8') if isinstance(doc.text, bytes) else doc.text 
                      for doc in results.docs if float(doc.vector_score) <= threshold]
        
        # --- STEP 3: Build the Augmented Prompt ---
        context_string = "\n".join(f"- {text}" for text in valid_docs) if valid_docs else "No specific local context found."
        
        prompt = f"""
        You are a helpful assistant. Use the Chat History and Local Context below to answer the user.
        
        [CHAT HISTORY]
        {history_context if history_context else "No previous conversation."}
        
        [LOCAL CONTEXT]
        {context_string}
        
        User Question: {query_text}
        Assistant Answer:
        """

        # --- STEP 4: Call Gemini ---
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        answer = response.text

        # --- STEP 5: Save this interaction to History ---
        # Store as JSON strings in a Redis List
        r.lpush(history_key, json.dumps({"role": "User", "content": query_text}))
        r.lpush(history_key, json.dumps({"role": "Assistant", "content": answer}))
        
        # Keep history manageable (last 20 entries = 10 rounds of conversation)
        r.ltrim(history_key, 0, 19)

        return jsonify({
            "question": query_text,
            "answer": answer,
            "source_count": len(valid_docs),
            "session_id": session_id,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "failure"}), 500

# --- 4. Run the Server ---
if __name__ == '__main__':
    # Running on 8080 as per your original script
    app.run(host='0.0.0.0', port=8080, debug=False)
