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

# Define Gradio API endpoint (no UI)
app = gr.Interface(
    fn=embed_text,
    inputs=gr.Textbox(label="Text Input"),
    outputs=gr.JSON(label="Embedding"),
    title="News Swipe Embedder",
    description="Embeds text using MiniLM for the News Swipe project."
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
