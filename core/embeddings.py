from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Global variable to store the model (load once, use many times)
_embedding_model = None

def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    """
    Load the sentence transformer model for creating embeddings
    Optimized for Mac M4 with MPS support
    
    Args:
        model_name (str): Name of the sentence transformer model
    
    Returns:
        SentenceTransformer: Loaded model
    """
    global _embedding_model
    
    if _embedding_model is not None:
        logger.info("Using cached embedding model")
        return _embedding_model
    
    try:
        logger.info(f"Loading embedding model: {model_name}")
        
        # Check if MPS (Metal Performance Shaders) is available on Mac M4
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Mac M4 Metal Performance Shaders (MPS) for acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA for acceleration")
        else:
            device = "cpu"
            logger.info("Using CPU for embedding model")
        
        # Load the model
        _embedding_model = SentenceTransformer(model_name, device=device)
        
        logger.info(f"Successfully loaded {model_name} on {device}")
        return _embedding_model
        
    except Exception as e:
        error_msg = f"Error loading embedding model: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def create_embeddings(texts):
    """
    Convert text chunks into vector embeddings
    
    Args:
        texts (list or str): Text chunks to convert to embeddings
    
    Returns:
        numpy.ndarray: Array of embeddings
    """
    try:
        # Load model if not already loaded
        model = load_embedding_model()
        
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        logger.info(f"Creating embeddings for {len(texts)} text chunks")
        
        # Create embeddings
        embeddings = model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10  # Show progress for large batches
        )
        
        logger.info(f"Successfully created embeddings with shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        error_msg = f"Error creating embeddings: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def get_text_embedding(text):
    """
    Get embedding for a single text (useful for user queries)
    
    Args:
        text (str): Single text to convert to embedding
    
    Returns:
        numpy.ndarray: Single embedding vector
    """
    try:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        embeddings = create_embeddings([text.strip()])
        return embeddings[0]  # Return single vector
        
    except Exception as e:
        error_msg = f"Error creating text embedding: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)