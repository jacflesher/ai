import os
import redis
import numpy as np
from flask import Flask, request, jsonify
from google import genai
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query


# Environment Validation
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

# Enviroment Configuration
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASS = os.getenv("REDIS_PASS")
INDEX_NAME = os.getenv("INDEX_NAME")
VECTOR_DIM = os.getenv("VECTOR_DIM")
MODEL_ID = os.getenv("MODEL_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setup & Configuration
app = Flask(__name__)


# Initialize Shared Resources (Loaded once on startup)
print("--- Initializing LLM Services ---")

# Gemini Client
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# Redis Connection
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=False)

# Embedding Model
print("Loading SentenceTransformer model (this may take a moment)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("--- Services Ready ---")

# --- 2. The Controller Endpoint ---

@app.route('/ask', methods=['POST'])
def ask_question():
    # Get data from the POST request
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    query_text = data.get('question')
    k = data.get('k', 4)
    threshold = data.get('threshold', 0.55)

    try:
        # Step 1: Vector Search in Redis
        query_embedding = embedding_model.encode(query_text)
        query_vector_bytes = query_embedding.astype(np.float32).tobytes()

        knn_query_string = f"*=>[KNN {k} @vector $vec_param AS vector_score]"
        q = Query(knn_query_string).sort_by("vector_score").paging(0, k).dialect(2)
        
        results = r.ft(INDEX_NAME).search(q, query_params={"vec_param": query_vector_bytes})
        
        # Filter by threshold (lower score is better for COSINE/L2)
        valid_docs = [doc.text for doc in results.docs if float(doc.vector_score) <= threshold]
        
        # Step 2: Build the Hybrid Prompt
        if valid_docs:
            context_string = "\n".join(f"- {text}" for text in valid_docs)
            prompt = f"""
            You are a helpful assistant. I have found some local context.
            1. Use the context below if it is relevant.
            2. If not, use your general knowledge.
            
            Context:
            {context_string}
            
            Question: {query_text}
            """
        else:
            prompt = f"Please answer the following question: {query_text}"

        # Step 3: Call Gemini 2.5 Flash
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        
        # Return the JSON response
        return jsonify({
            "question": query_text,
            "answer": response.text,
            "source_count": len(valid_docs),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "failure"}), 500

# --- 3. Run the Server ---
if __name__ == '__main__':
    # Run on port 5000 (Flask default)
    app.run(host='0.0.0.0', port=8080, debug=False)
