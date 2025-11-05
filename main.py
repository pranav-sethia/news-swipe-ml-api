# ----- IMPORTS

import torch
from fastapi import FastAPI 
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer 
from fastapi.middleware.cors import CORSMiddleware

# ----- LOAD MODEL 

# Check if GPU is available and set device accordingly
try:
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
except exception as e:
    print(f"Error checking device - defaulting to CPU: {e}")
    device = "cpu"

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
print("Model loaded successfully.")

# ----- FASTAPI SETUP
app = FastAPI()

# ----- CORS CONFIGURATION

# Allows React app to communicate with this FastAPI backend
app.add_middlewear(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- DEFINE REQUEST BODY 

# Tells FastAPI what to expect in the request body
class TextItem(BaseModel):
    text: str

# ----- CREATE /embed ENDPOINT

@app.post("/embed")
def create_embeddings(item: TextItem):
    """
    Receives and returns 384 dimensional text embedding
    """

    # Prevent crashing from overly long text
    if (len(item.text) > 2000):
        return {"error": "Input text is too long. Maximum length is 2000 characters."}, 400
    
    try:
        embedding = model.encode(item.text).tolist() # Converts numpy array to Python list for JSON
        return {"embedding": embedding}
    except Exception as e:
        return {"error": f"Failed to generate embedding: {e}"}, 500


