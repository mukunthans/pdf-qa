import PyPDF2
import io
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from uploaded PDF file
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        str: Extracted text content
    """
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        
        # Check if PDF is encrypted
        if pdf_reader.is_encrypted:
            logger.warning("PDF is encrypted. Attempting to decrypt...")
            try:
                pdf_reader.decrypt('')  # Try empty password
            except:
                return "Error: PDF is password protected. Please provide an unlocked PDF."
        
        # Extract text from all pages
        full_text = ""
        total_pages = len(pdf_reader.pages)
        
        logger.info(f"Processing PDF with {total_pages} pages...")
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():  # Only add if page has text
                    full_text += page_text + "\n\n"
                    logger.info(f"Extracted text from page {page_num + 1}")
                else:
                    logger.warning(f"Page {page_num + 1} appears to be empty or image-only")
            except Exception as e:
                logger.error(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue
        
        # Clean up the extracted text
        cleaned_text = clean_extracted_text(full_text)
        
        if not cleaned_text.strip():
            return "Error: No readable text found in PDF. This might be a scanned document or image-only PDF."
        
        logger.info(f"Successfully extracted {len(cleaned_text)} characters from PDF")
        return cleaned_text
        
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        logger.error(error_msg)
        return error_msg

def clean_extracted_text(text):
    """
    Clean and normalize extracted text
    
    Args:
        text (str): Raw extracted text
    
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Strip whitespace and skip empty lines
        cleaned_line = line.strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    # Join with single spaces and normalize
    cleaned_text = ' '.join(cleaned_lines)
    
    # Remove multiple spaces
    import re
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()

import re

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into overlapping chunks for better context preservation
    
    Args:
        text (str): Text to be chunked
        chunk_size (int): Target size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks
    
    Returns:
        list: List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Clean the text first
    text = text.strip()
    
    # If text is shorter than chunk_size, return as single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If this is the last chunk, take all remaining text
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        
        # Try to break at sentence boundary
        chunk_text = text[start:end]
        
        # Find the last sentence ending within the chunk
        sentence_endings = ['.', '!', '?', '\n']
        best_break = -1
        
        for i in range(len(chunk_text) - 1, max(len(chunk_text) - 100, 0), -1):
            if chunk_text[i] in sentence_endings and i < len(chunk_text) - 1:
                if chunk_text[i + 1].isspace() or chunk_text[i + 1].isupper():
                    best_break = start + i + 1
                    break
        
        # If we found a good sentence break, use it
        if best_break > start:
            chunk = text[start:best_break].strip()
        else:
            # Otherwise, try to break at word boundary
            chunk_text = text[start:end]
            last_space = chunk_text.rfind(' ')
            if last_space > chunk_size * 0.7:  # Only if we don't lose too much
                chunk = text[start:start + last_space].strip()
                end = start + last_space
            else:
                chunk = chunk_text.strip()
        
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap
        
        # Make sure we don't go backwards
        if start < 0:
            start = 0
    
    # Remove any empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks

def get_chunk_info(chunks):
    """
    Get information about the chunks for debugging/monitoring
    
    Args:
        chunks (list): List of text chunks
    
    Returns:
        dict: Information about chunks
    """
    if not chunks:
        return {"total_chunks": 0, "avg_length": 0, "total_chars": 0}
    
    total_chars = sum(len(chunk) for chunk in chunks)
    avg_length = total_chars / len(chunks)
    
    return {
        "total_chunks": len(chunks),
        "avg_length": round(avg_length, 1),
        "total_chars": total_chars,
        "shortest_chunk": min(len(chunk) for chunk in chunks),
        "longest_chunk": max(len(chunk) for chunk in chunks)
    }