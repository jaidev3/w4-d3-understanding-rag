import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    NLTKTextSplitter,
)
from sentence_transformers import SentenceTransformer
import tiktoken
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Ensure both legacy and new Punkt resources are available
for _dir in ("tokenizers/punkt", "tokenizers/punkt_tab"):
    try:
        nltk.data.find(_dir)
    except LookupError:
        # Determine which corpus name to download based on the missing directory
        _corpus = "punkt_tab" if "punkt_tab" in _dir else "punkt"
        try:
            nltk.download(_corpus)
        except Exception:
            # Fallback for environments where 'punkt_tab' is not explicitly
            # available as a separate corpus (older NLTK versions).
            nltk.download("punkt")

class ChunkingStrategies:
    """Collection of different chunking strategies for RAG systems"""
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def count_tokens(self, text):
        """Count tokens using tiktoken"""
        return len(self.encoding.encode(text))
    
    def fixed_size_chunking(self, text, chunk_size=1000, overlap=200):
        """Fixed size chunking with overlap"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                'chunk_id': i,
                'content': chunk,
                'char_count': len(chunk),
                'token_count': self.count_tokens(chunk),
                'word_count': len(chunk.split()),
                'strategy': 'Fixed Size'
            })
        return chunk_data
    
    def sentence_based_chunking(self, text, max_sentences=5):
        """Sentence-based chunking"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= max_sentences:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
        
        # Add remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                'chunk_id': i,
                'content': chunk,
                'char_count': len(chunk),
                'token_count': self.count_tokens(chunk),
                'word_count': len(chunk.split()),
                'strategy': 'Sentence-Based'
            })
        return chunk_data
    
    def paragraph_based_chunking(self, text):
        """Paragraph-based chunking"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunk_data = []
        for i, paragraph in enumerate(paragraphs):
            if paragraph:  # Only non-empty paragraphs
                chunk_data.append({
                    'chunk_id': i,
                    'content': paragraph,
                    'char_count': len(paragraph),
                    'token_count': self.count_tokens(paragraph),
                    'word_count': len(paragraph.split()),
                    'strategy': 'Paragraph-Based'
                })
        return chunk_data
    
    def semantic_chunking(self, text, similarity_threshold=0.7):
        """Semantic chunking using sentence transformers"""
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            sentences = sent_tokenize(text)
            
            if len(sentences) < 2:
                return [{
                    'chunk_id': 0,
                    'content': text,
                    'char_count': len(text),
                    'token_count': self.count_tokens(text),
                    'word_count': len(text.split()),
                    'strategy': 'Semantic'
                }]
            
            embeddings = model.encode(sentences)
            
            chunks = []
            current_chunk = [sentences[0]]
            
            for i in range(1, len(sentences)):
                similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                    np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
                )
                
                if similarity >= similarity_threshold:
                    current_chunk.append(sentences[i])
                else:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentences[i]]
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            chunk_data = []
            for i, chunk in enumerate(chunks):
                chunk_data.append({
                    'chunk_id': i,
                    'content': chunk,
                    'char_count': len(chunk),
                    'token_count': self.count_tokens(chunk),
                    'word_count': len(chunk.split()),
                    'strategy': 'Semantic'
                })
            return chunk_data
            
        except Exception as e:
            st.error(f"Error in semantic chunking: {str(e)}")
            return self.fixed_size_chunking(text)
    
    def token_based_chunking(self, text, chunk_size=512, overlap=50):
        """Token-based chunking using tiktoken"""
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            encoding_name="cl100k_base"
        )
        chunks = splitter.split_text(text)
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                'chunk_id': i,
                'content': chunk,
                'char_count': len(chunk),
                'token_count': self.count_tokens(chunk),
                'word_count': len(chunk.split()),
                'strategy': 'Token-Based'
            })
        return chunk_data
    
    def recursive_character_chunking(self, text, chunk_size=1000, overlap=200):
        """Recursive character text splitting"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                'chunk_id': i,
                'content': chunk,
                'char_count': len(chunk),
                'token_count': self.count_tokens(chunk),
                'word_count': len(chunk.split()),
                'strategy': 'Recursive Character'
            })
        return chunk_data
    
    def sliding_window_chunking(self, text, window_size=1000, step_size=500):
        """Sliding window chunking"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + window_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == len(text):
                break
            start += step_size
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                'chunk_id': i,
                'content': chunk,
                'char_count': len(chunk),
                'token_count': self.count_tokens(chunk),
                'word_count': len(chunk.split()),
                'strategy': 'Sliding Window'
            })
        return chunk_data
    
    def apply_strategy(self, strategy, text, params):
        """Apply the selected chunking strategy with parameters"""
        if strategy == 'Fixed Size':
            return self.fixed_size_chunking(
                text, 
                params.get('chunk_size', 1000), 
                params.get('overlap', 200)
            )
        elif strategy == 'Sentence-Based':
            return self.sentence_based_chunking(
                text, 
                params.get('max_sentences', 5)
            )
        elif strategy == 'Paragraph-Based':
            return self.paragraph_based_chunking(text)
        elif strategy == 'Semantic':
            return self.semantic_chunking(
                text, 
                params.get('similarity_threshold', 0.7)
            )
        elif strategy == 'Token-Based':
            return self.token_based_chunking(
                text, 
                params.get('chunk_size', 512), 
                params.get('overlap', 50)
            )
        elif strategy == 'Recursive Character':
            return self.recursive_character_chunking(
                text, 
                params.get('chunk_size', 1000), 
                params.get('overlap', 200)
            )
        elif strategy == 'Sliding Window':
            return self.sliding_window_chunking(
                text, 
                params.get('window_size', 1000), 
                params.get('step_size', 500)
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

def get_available_strategies():
    """Get list of available chunking strategies"""
    return [
        'Fixed Size',
        'Sentence-Based',
        'Paragraph-Based',
        'Semantic',
        'Token-Based',
        'Recursive Character',
        'Sliding Window'
    ]

def get_strategy_info():
    """Get information about each chunking strategy"""
    return {
        'Fixed Size': {
            'description': 'Splits text into chunks of fixed character size with configurable overlap.',
            'use_cases': 'General purpose, consistent chunk sizes for embedding models',
            'pros': 'Predictable chunk sizes, simple implementation',
            'cons': 'May break sentences or paragraphs unnaturally',
            'parameters': ['chunk_size', 'overlap']
        },
        'Sentence-Based': {
            'description': 'Groups sentences together into chunks based on sentence boundaries.',
            'use_cases': 'When semantic coherence at sentence level is important',
            'pros': 'Preserves sentence integrity, natural language boundaries',
            'cons': 'Variable chunk sizes, may create very small or large chunks',
            'parameters': ['max_sentences']
        },
        'Paragraph-Based': {
            'description': 'Uses paragraph breaks as natural chunk boundaries.',
            'use_cases': 'Documents with clear paragraph structure',
            'pros': 'Maintains document structure, coherent content blocks',
            'cons': 'Highly variable chunk sizes, depends on document formatting',
            'parameters': []
        },
        'Semantic': {
            'description': 'Uses sentence similarity to group semantically related content.',
            'use_cases': 'When semantic coherence is critical for retrieval',
            'pros': 'Maintains semantic coherence, intelligent content grouping',
            'cons': 'Computationally expensive, requires embedding model',
            'parameters': ['similarity_threshold']
        },
        'Token-Based': {
            'description': 'Splits text based on token count using tiktoken encoding.',
            'use_cases': 'When working with LLMs that have token limits',
            'pros': 'Precise token control, optimal for LLM processing',
            'cons': 'May break words or sentences at token boundaries',
            'parameters': ['chunk_size', 'overlap']
        },
        'Recursive Character': {
            'description': 'Recursively splits text using hierarchical separators.',
            'use_cases': 'Balanced approach for most document types',
            'pros': 'Intelligent splitting, preserves structure when possible',
            'cons': 'More complex logic, may still break semantic units',
            'parameters': ['chunk_size', 'overlap']
        },
        'Sliding Window': {
            'description': 'Creates overlapping chunks using a sliding window approach.',
            'use_cases': 'When context overlap is crucial for retrieval',
            'pros': 'High overlap ensures context preservation',
            'cons': 'Creates more chunks, potential information redundancy',
            'parameters': ['window_size', 'step_size']
        }
    } 