import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def generate_gemini_response(query, context):
    """
    Generate response using Gemini API based on context and query
    
    Args:
        query (str): User's question
        context (str): Relevant context from PDF
    
    Returns:
        dict: Response with answer and metadata
    """
    try:
        if not query.strip():
            return {
                "answer": "Please provide a valid question.",
                "status": "error"
            }
        
        if not context.strip():
            return {
                "answer": "I don't have enough relevant information in the document to answer your question.",
                "status": "no_context"
            }
        
        # Create the prompt for Gemini
        prompt = f"""
You are a helpful assistant that answers questions based on provided document content.

CONTEXT FROM DOCUMENT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Be specific and cite relevant parts when possible
4. Keep your answer concise but complete
5. If asked about something not in the context, state that the information is not available in the document

ANSWER:"""

        logger.info("Sending request to Gemini API")
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response
        response = model.generate_content(prompt)
        
        if response and response.text:
            answer = response.text.strip()
            logger.info("Successfully received response from Gemini")
            
            return {
                "answer": answer,
                "status": "success",
                "model_used": "gemini-1.5-flash"
            }
        else:
            return {
                "answer": "I apologize, but I couldn't generate a response. Please try again.",
                "status": "empty_response"
            }
            
    except Exception as e:
        error_msg = f"Error generating Gemini response: {str(e)}"
        logger.error(error_msg)
        
        # Handle specific API errors
        if "API_KEY" in str(e):
            return {
                "answer": "API key error. Please check your Gemini API configuration.",
                "status": "api_key_error"
            }
        elif "quota" in str(e).lower():
            return {
                "answer": "API quota exceeded. Please try again later.",
                "status": "quota_error"
            }
        else:
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "status": "error"
            }