import streamlit as st
from utils import answer_question
from core.pdf_handler import extract_text_from_pdf, chunk_text
from core.embeddings import load_embedding_model, create_embeddings
from core.vector_store import (
    create_vector_store, 
    clear_vector_store, 
    get_vector_store_info
)
import config

st.title("PDF Q&A System - Complete Pipeline")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Process PDF
    clear_vector_store()
    
    with st.spinner("Processing PDF..."):
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(uploaded_file)
        
        # Chunk the text
        chunks = chunk_text(
            extracted_text, 
            chunk_size=config.CHUNK_SIZE, 
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        # Load embedding model
        model = load_embedding_model()
        
        # Create embeddings for chunks
        embeddings = create_embeddings(chunks)
        
        # Create vector store
        index, embeddings, document_chunks = create_vector_store(chunks, embeddings)
    
    st.success("PDF processed successfully!")
    
    # Show document info
    store_info = get_vector_store_info()
    st.write(f"Document loaded: {store_info['total_chunks']} chunks created")
    
    # Q&A Interface
    st.subheader("Ask Questions About Your Document")
    
    query = st.text_input("Enter your question:", placeholder="What is this document about?")
    
    if query:
        with st.spinner("Finding answer..."):
            # Get complete answer
            result = answer_question(query)
        
        # Display answer
        st.subheader("Answer:")
        
        if result["status"] == "success":
            st.success("✅ Answer generated successfully")
            st.write(result["answer"])
            
            # Show context used
            if result["context_chunks"]:
                with st.expander("📄 View source context"):
                    for i, (chunk, score) in enumerate(result["context_chunks"]):
                        st.write(f"**Source {i+1}** (Relevance: {score:.2f})")
                        st.text_area(f"Context {i+1}", chunk, height=100, key=f"context_{i}")
        
        elif result["status"] == "no_context":
            st.warning("⚠️ No relevant information found")
            st.write(result["answer"])
        
        else:
            st.error("❌ Error occurred")
            st.write(result["answer"])

# Sidebar status
st.sidebar.subheader("System Status")
store_info = get_vector_store_info()
st.sidebar.write(f"Document: {store_info['status']}")
if store_info['status'] == 'ready':
    st.sidebar.write(f"Chunks: {store_info['total_chunks']}")