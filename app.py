import torch
from sentence_transformers import SentenceTransformer
import gradio as gr

# Lazy-load model
model = None

def get_model():
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    return model

def embed_text(text):
    if len(text) > 2000:
        return {"error": "Input text too long (max 2000 chars)"}
    model = get_model()
    embedding = model.encode(text).tolist()
    return {"embedding": embedding}

# --- API-only Gradio setup using Blocks ---
with gr.Blocks() as demo:
    gr.JSON(embed_text, label="Embedding")

if __name__ == "__main__":
    # Launch with server_name=0.0.0.0 so it's reachable from outside
    # show_api=True ensures /api/predict endpoint works
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True,
        allow_flagging="never",
        share=False  # or True if you want a public shareable link
    )
