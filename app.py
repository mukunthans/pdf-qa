import streamlit as st
from utils import extract_text_from_pdf, chunk_text, get_chunk_info

st.title("PDF Text Extractor & Chunker Test")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
    
    st.success("Text extraction complete!")
    
    # Show extraction stats
    st.subheader("Extraction Stats:")
    st.write(f"Total characters: {len(extracted_text)}")
    st.write(f"Total words: {len(extracted_text.split())}")
    
    # Now chunk the text
    with st.spinner("Chunking text..."):
        chunks = chunk_text(extracted_text)
        chunk_info = get_chunk_info(chunks)
    
    st.success("Text chunking complete!")
    
    # Show chunking stats
    st.subheader("Chunking Stats:")
    for key, value in chunk_info.items():
        st.write(f"{key.replace('_', ' ').title()}: {value}")
    
    # Show first few chunks
    st.subheader("Sample Chunks:")
    for i, chunk in enumerate(chunks[:3]):
        st.text_area(f"Chunk {i+1}", chunk[:300] + "..." if len(chunk) > 300 else chunk, height=100)