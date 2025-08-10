import os
import logging
from datetime import datetime
from core.embeddings import get_text_embedding
from core.vector_store import search_similar_chunks
from models.gemini import generate_gemini_response

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Define log file path (single file for all runs)
log_file_path = os.path.join("logs", "pdf_extraction.log")

# Configure logging to file with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
        logging.StreamHandler()  # keep console output
    ]
)

# Create logger object
logger = logging.getLogger(__name__)

# Mark the start of a new run
logger.info("========== New run started ==========")

def get_relevant_context(query, top_k=3):
    """
    Get relevant context chunks for a user query
    
    Args:
        query (str): User's question
        top_k (int): Number of chunks to retrieve
    
    Returns:
        dict: Context information with chunks and metadata
    """
    try:
        if not query or not query.strip():
            return {
                "context": "",
                "chunks": [],
                "message": "Please provide a valid question."
            }
        
        # Get query embedding
        query_embedding = get_text_embedding(query)
        
        # Search for similar chunks
        search_results = search_similar_chunks(query_embedding, top_k=top_k)
        
        if not search_results:
            return {
                "context": "",
                "chunks": [],
                "message": "No relevant content found in the document."
            }
        
        # Filter results by similarity threshold (0.3 is reasonable)
        relevant_results = [(chunk, score) for chunk, score in search_results if score > 0.3]
        
        if not relevant_results:
            return {
                "context": "",
                "chunks": [],
                "message": "No sufficiently relevant content found for your question."
            }
        
        # Combine chunks into context
        context_parts = []
        for i, (chunk, score) in enumerate(relevant_results, 1):
            context_parts.append(f"[Context {i}]:\n{chunk}\n")
        
        context = "\n".join(context_parts)
        
        logger.info(f"Retrieved {len(relevant_results)} relevant chunks for query")
        
        return {
            "context": context,
            "chunks": relevant_results,
            "message": f"Found {len(relevant_results)} relevant sections."
        }
        
    except Exception as e:
        error_msg = f"Error retrieving context: {str(e)}"
        logger.error(error_msg)
        return {
            "context": "",
            "chunks": [],
            "message": error_msg
        }

def answer_question(query):
    """
    Complete Q&A pipeline: retrieve context and generate answer
    
    Args:
        query (str): User's question
    
    Returns:
        dict: Complete response with answer, context, and metadata
    """
    try:
        logger.info(f"Processing question: '{query[:50]}...'")
        
        # Step 1: Get relevant context
        context_result = get_relevant_context(query)
        
        if not context_result["context"]:
            return {
                "answer": context_result["message"],
                "context_chunks": [],
                "status": "no_context",
                "query": query
            }
        
        # Step 2: Generate answer with Gemini
        gemini_result = generate_gemini_response(query, context_result["context"])
        
        return {
            "answer": gemini_result["answer"],
            "context_chunks": context_result["chunks"],
            "status": gemini_result["status"],
            "query": query,
            "model_used": gemini_result.get("model_used", "unknown")
        }
        
    except Exception as e:
        error_msg = f"Error in Q&A pipeline: {str(e)}"
        logger.error(error_msg)
        
        return {
            "answer": f"An error occurred while processing your question: {str(e)}",
            "context_chunks": [],
            "status": "error",
            "query": query
        }