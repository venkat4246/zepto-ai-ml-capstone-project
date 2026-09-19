
from sentence_transformers import SentenceTransformer

# Load MiniLM embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(texts):
    """Generate embeddings for a list of texts."""
    return embedding_model.encode(
        texts,
        show_progress_bar=False
    )
