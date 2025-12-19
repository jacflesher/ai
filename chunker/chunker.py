import sys
import os
import redis
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

# Environment Validation
errors = []
vars_to_check = [
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASS", "INDEX_NAME", "VECTOR_DIM"
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

# Initialize Redis and Embedding Model 
try:
    # Note: decode_responses=False is REQUIRED when storing/retrieving binary vectors
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=False)
    r.ping()
    print("Connected to Redis.")
except Exception as e:
    print(f"Redis Error: {e}")
    exit()

def create_index_if_missing():
    try:
        # Check if index exists
        r.ft(INDEX_NAME).info()
        print(f"Index '{INDEX_NAME}' already exists.")
    except:
        print(f"Index '{INDEX_NAME}' missing. Creating now...")
        schema = (
            TextField("text"),
            VectorField("vector", "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIM,
                "DISTANCE_METRIC": "COSINE"
            })
        )
        # Prefix tells Redis to look for keys starting with "doc:"
        definition = IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
        
        r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
        print(f"Index '{INDEX_NAME}' created successfully.")

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and Split the Input File
if len(sys.argv) < 2:
    print("Usage: python ingest.py <path_to_file>")
    exit()

input_file = Path(sys.argv[1])
if not input_file.exists():
    print(f"Error: File {input_file} not found.")
    exit()

print(f"Processing: {input_file.name}")
loader = TextLoader(str(input_file), encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)
chunks = splitter.split_documents(docs)
print(f"Split document into {len(chunks)} chunks.")

# Process and Upload to Redis
# Ensure index exists before uploading
create_index_if_missing()

print(f"Generating embeddings for {len(chunks)} chunks (Batch mode)...")
# OPTIMIZATION: Encode everything in one go instead of inside the loop
texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(texts)

print(f"Uploading to Redis index '{INDEX_NAME}'...")
pipeline = r.pipeline()
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    # Unique key: doc:filename:timestamp:chunk_id
    doc_key = f"doc:{input_file.stem}:{timestamp}:{i:03d}"
    
    pipeline.hset(doc_key, mapping={
        "text": text,
        "vector": embedding.astype(np.float32).tobytes(),
        "source": str(input_file.name),
        "chunk_id": i
    })

pipeline.execute()
print(f"Successfully uploaded {len(chunks)} chunks to Redis.")

# Optional: Save local backup 
output_dir = Path("chunks")
output_dir.mkdir(exist_ok=True)
for i, text in enumerate(texts):
    out_file = output_dir / f"{timestamp}_{input_file.stem}_chunk-{i:03d}.txt"
    out_file.write_text(text, encoding="utf-8")
print(f"Local text backups saved to {output_dir}/")
