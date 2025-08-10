import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Global variables for session storage
_vector_store = None
_document_chunks = None

def create_vector_store(chunks, embeddings):
    """
    Create FAISS vector store from document chunks
    
    Args:
        chunks (list): List of text chunks from PDF
        embeddings (numpy.ndarray): Embeddings for the chunks
    
    Returns:
        tuple: (faiss_index, embeddings, chunks)
    """
    global _vector_store, _document_chunks
    
    try:
        logger.info(f"Creating vector store for {len(chunks)} chunks")
        
        # Get embedding dimension
        dimension = embeddings.shape[1]  # Should be 384 for all-MiniLM-L6-v2
        
        # Create FAISS index (IndexFlatIP for cosine similarity)
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        # Store in global variables for session
        _vector_store = index
        _document_chunks = chunks
        
        logger.info(f"Vector store created successfully with {index.ntotal} vectors")
        return index, embeddings, chunks
        
    except Exception as e:
        error_msg = f"Error creating vector store: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def search_similar_chunks(query_embedding, top_k=3):
    """
    Search for similar chunks in the vector store
    
    Args:
        query_embedding (numpy.ndarray): Embedding of the user's question
        top_k (int): Number of similar chunks to return
    
    Returns:
        list: List of tuples (chunk_text, similarity_score)
    """
    global _vector_store, _document_chunks
    
    try:
        if _vector_store is None or _document_chunks is None:
            raise ValueError("Vector store not initialized. Please upload a PDF first.")
        
        logger.info(f"Searching for {top_k} similar chunks")
        
        # Reshape and prepare query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        similarities, indices = _vector_store.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(_document_chunks):  # Safety check
                chunk_text = _document_chunks[idx]
                results.append((chunk_text, float(similarity)))
                logger.info(f"Found similar chunk {i+1} with similarity: {similarity:.3f}")
        
        logger.info(f"Retrieved {len(results)} similar chunks")
        return results
        
    except Exception as e:
        error_msg = f"Error searching similar chunks: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def clear_vector_store():
    """
    Clear the current vector store (for new PDF uploads)
    """
    global _vector_store, _document_chunks
    
    _vector_store = None
    _document_chunks = None
    logger.info("Vector store cleared")

def get_vector_store_info():
    """
    Get information about current vector store
    
    Returns:
        dict: Information about the vector store
    """
    global _vector_store, _document_chunks
    
    if _vector_store is None:
        return {"status": "empty", "total_vectors": 0, "total_chunks": 0}
    
    return {
        "status": "ready",
        "total_vectors": _vector_store.ntotal,
        "total_chunks": len(_document_chunks) if _document_chunks else 0,
        "dimension": _vector_store.d
    }