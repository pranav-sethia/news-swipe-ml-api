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

# --- API-only Gradio setup ---
# Use Interface but no UI
iface = gr.Interface(
    fn=embed_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here"),
    outputs=gr.JSON(),
    allow_flagging="never",
    title="News Swipe Embedder",
    description="Embeds text using MiniLM for the News Swipe project."
)

if __name__ == "__main__":
    # Hide the UI but enable API
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True,  # enables /api/predict
        inbrowser=False,
        share=False
    )
