from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load model
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class TextItem(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    global model
    print("Model will load on first request, not at startup.")

@app.post("/embed")
def create_embeddings(item: TextItem):
    global model
    if model is None:
        print("⏳ Loading model for the first time...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        print("✅ Model loaded successfully.")

    embedding = model.encode(item.text).tolist()
    return {"embedding": embedding}

@app.get("/")
def root():
    return {"status": "Running (model loads lazily)"}
