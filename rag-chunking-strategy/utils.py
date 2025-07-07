import io
import re
from pypdf import PdfReader
import streamlit as st

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = text.replace('\x00', '')
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def validate_pdf_file(uploaded_file):
    """Validate that the uploaded file is a valid PDF"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "File must be a PDF"
    
    # Check file size (limit to 50MB)
    if uploaded_file.size > 50 * 1024 * 1024:
        return False, "File size must be less than 50MB"
    
    return True, "Valid PDF file"

def calculate_text_metrics(text):
    """Calculate various text metrics"""
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'paragraph_count': 0,
            'sentence_count': 0
        }
    
    char_count = len(text)
    word_count = len(text.split())
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    
    # Simple sentence count (approximation)
    sentence_count = len([s for s in text.split('.') if s.strip()])
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'paragraph_count': paragraph_count,
        'sentence_count': sentence_count
    }

def format_number(num):
    """Format number with commas for better readability"""
    return f"{num:,}"

def truncate_text(text, max_length=100):
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def get_file_info(uploaded_file):
    """Get information about the uploaded file"""
    if uploaded_file is None:
        return {}
    
    return {
        'name': uploaded_file.name,
        'size': uploaded_file.size,
        'type': uploaded_file.type,
        'size_mb': round(uploaded_file.size / (1024 * 1024), 2)
    }

def safe_divide(numerator, denominator):
    """Safely divide two numbers, returning 0 if denominator is 0"""
    if denominator == 0:
        return 0
    return numerator / denominator

def chunk_overlap_analysis(chunks):
    """Analyze overlap between consecutive chunks"""
    if len(chunks) < 2:
        return []
    
    overlaps = []
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]['content']
        next_chunk = chunks[i + 1]['content']
        
        # Simple overlap detection (find common substrings)
        overlap_chars = 0
        min_length = min(len(current_chunk), len(next_chunk))
        
        # Check for overlap at the end of current chunk and start of next chunk
        for j in range(1, min_length + 1):
            if current_chunk[-j:] == next_chunk[:j]:
                overlap_chars = j
        
        overlaps.append({
            'chunk_pair': f"{i}-{i+1}",
            'overlap_chars': overlap_chars,
            'overlap_percentage': (overlap_chars / min_length) * 100 if min_length > 0 else 0
        })
    
    return overlaps

def estimate_processing_time(text_length, strategy):
    """Estimate processing time based on text length and strategy"""
    # Base time estimates in seconds
    base_times = {
        'Fixed Size': 0.001,
        'Sentence-Based': 0.002,
        'Paragraph-Based': 0.001,
        'Semantic': 0.1,  # Much slower due to embedding computation
        'Token-Based': 0.003,
        'Recursive Character': 0.002,
        'Sliding Window': 0.001
    }
    
    base_time = base_times.get(strategy, 0.002)
    estimated_time = base_time * (text_length / 1000)  # Scale with text length
    
    return max(estimated_time, 0.1)  # Minimum 0.1 seconds

def get_chunk_size_recommendations(text_length):
    """Get chunk size recommendations based on text length"""
    if text_length < 1000:
        return {
            'small': 100,
            'medium': 200,
            'large': 500
        }
    elif text_length < 10000:
        return {
            'small': 500,
            'medium': 1000,
            'large': 2000
        }
    else:
        return {
            'small': 1000,
            'medium': 2000,
            'large': 4000
        }

def validate_chunk_parameters(strategy, params):
    """Validate chunk parameters for a given strategy"""
    errors = []
    
    if strategy in ['Fixed Size', 'Recursive Character']:
        chunk_size = params.get('chunk_size', 1000)
        overlap = params.get('overlap', 200)
        
        if chunk_size <= 0:
            errors.append("Chunk size must be positive")
        if overlap < 0:
            errors.append("Overlap cannot be negative")
        if overlap >= chunk_size:
            errors.append("Overlap must be less than chunk size")
    
    elif strategy == 'Token-Based':
        chunk_size = params.get('chunk_size', 512)
        overlap = params.get('overlap', 50)
        
        if chunk_size <= 0:
            errors.append("Chunk size must be positive")
        if overlap < 0:
            errors.append("Overlap cannot be negative")
        if overlap >= chunk_size:
            errors.append("Overlap must be less than chunk size")
    
    elif strategy == 'Sentence-Based':
        max_sentences = params.get('max_sentences', 5)
        
        if max_sentences <= 0:
            errors.append("Max sentences must be positive")
    
    elif strategy == 'Semantic':
        similarity_threshold = params.get('similarity_threshold', 0.7)
        
        if not (0 <= similarity_threshold <= 1):
            errors.append("Similarity threshold must be between 0 and 1")
    
    elif strategy == 'Sliding Window':
        window_size = params.get('window_size', 1000)
        step_size = params.get('step_size', 500)
        
        if window_size <= 0:
            errors.append("Window size must be positive")
        if step_size <= 0:
            errors.append("Step size must be positive")
        if step_size >= window_size:
            errors.append("Step size should be less than window size for meaningful overlap")
    
    return errors 