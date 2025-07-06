import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import os

# Set page config
st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    return OpenAI(api_key=api_key)

# Initialize sentence transformer model
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# Generate embeddings using Sentence-Transformers
@st.cache_data
def generate_embeddings(texts):
    model = load_sentence_model()
    embeddings = model.encode(texts)
    return embeddings

# Generate embeddings using OpenAI API
@st.cache_data
def generate_openai_embeddings(texts):
    client = get_openai_client()
    embeddings = []
    progress_bar = st.progress(0)
    for i, text in enumerate(texts):
        try:
            response = client.embeddings.create(input=text, model="text-embedding-ada-002")
            embeddings.append(response.data[0].embedding)
            progress_bar.progress((i + 1) / len(texts))
        except Exception as e:
            st.error(f"Error generating OpenAI embedding for text {i+1}: {str(e)}")
            return None
    progress_bar.empty()
    return embeddings

# Cosine similarity calculation
def calculate_similarity(embeddings):
    return cosine_similarity(embeddings)

# Detect clones based on a similarity threshold
def detect_clones(similarity_matrix, threshold=0.8):
    clones = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):  # Avoid double counting
            if similarity_matrix[i][j] >= threshold:
                clones.append((i, j, similarity_matrix[i][j]))
    return clones

# Create similarity matrix visualization
def plot_similarity_matrix(similarity_matrix, title="Similarity Matrix"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', fmt=".3f", ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Text Index')
    ax.set_ylabel('Text Index')
    return fig

# Main Streamlit app
def main():
    st.title("🔍 Plagiarism Detector - Semantic Similarity")
    st.markdown("Compare multiple texts using advanced AI models to detect potential plagiarism and semantic similarity.")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        similarity_threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.8, 
            step=0.05,
            help="Texts with similarity above this threshold will be flagged as potential clones"
        )
        
        use_openai = st.checkbox(
            "Use OpenAI Embeddings", 
            value=False,
            help="Enable OpenAI embeddings (requires API key)"
        )
        
        preprocess_texts = st.checkbox(
            "Preprocess Texts", 
            value=True,
            help="Apply text preprocessing (lowercase, remove punctuation, etc.)"
        )
    
    # Text input section
    st.header("📝 Input Texts")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Manual Input", "File Upload"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            text1 = st.text_area("Text 1", height=100, placeholder="Enter first text here...")
            text2 = st.text_area("Text 2", height=100, placeholder="Enter second text here...")
            text3 = st.text_area("Text 3", height=100, placeholder="Enter third text here...")
        
        with col2:
            text4 = st.text_area("Text 4", height=100, placeholder="Enter fourth text here...")
            text5 = st.text_area("Text 5", height=100, placeholder="Enter fifth text here...")
    
    with tab2:
        st.info("📁 Upload text files for comparison")
        uploaded_files = st.file_uploader(
            "Choose text files", 
            accept_multiple_files=True, 
            type=['txt'],
            help="Upload up to 5 text files"
        )
        
        if uploaded_files:
            texts_from_files = []
            for file in uploaded_files[:5]:  # Limit to 5 files
                content = file.read().decode('utf-8')
                texts_from_files.append(content)
            
            # Display uploaded file contents
            for i, (file, content) in enumerate(zip(uploaded_files[:5], texts_from_files)):
                with st.expander(f"📄 {file.name}"):
                    st.text_area(f"Content", value=content, height=100, key=f"file_{i}")
    
    # Process texts
    if st.button("🔍 Analyze Texts", type="primary"):
        # Collect texts
        if 'texts_from_files' in locals() and texts_from_files:
            texts = texts_from_files
        else:
            texts = [text1, text2, text3, text4, text5]
        
        # Filter out empty texts
        texts = [text for text in texts if text.strip()]
        
        if len(texts) < 2:
            st.error("Please provide at least 2 texts for comparison.")
            return
        
        # Preprocess texts if enabled
        if preprocess_texts:
            processed_texts = [preprocess_text(text) for text in texts]
        else:
            processed_texts = texts
        
        # Show processing status
        with st.spinner("Processing texts..."):
            # Generate embeddings using Sentence Transformers
            st.info("🤖 Generating embeddings using Sentence Transformers...")
            sentence_embeddings = generate_embeddings(processed_texts)
            sentence_sim = calculate_similarity(sentence_embeddings)
            sentence_clones = detect_clones(sentence_sim, threshold=similarity_threshold)
            
            # Generate OpenAI embeddings if enabled
            openai_embeddings = None
            openai_sim = None
            openai_clones = None
            
            if use_openai:
                st.info("🧠 Generating embeddings using OpenAI...")
                openai_embeddings = generate_openai_embeddings(processed_texts)
                if openai_embeddings:
                    openai_sim = calculate_similarity(openai_embeddings)
                    openai_clones = detect_clones(openai_sim, threshold=similarity_threshold)
        
        # Display results
        st.header("📊 Results")
        
        # Create tabs for results
        if use_openai and openai_embeddings:
            result_tabs = st.tabs(["Sentence Transformers", "OpenAI Embeddings", "Comparison"])
        else:
            result_tabs = st.tabs(["Sentence Transformers"])
        
        with result_tabs[0]:
            st.subheader("🤖 Sentence Transformers Results")
            
            # Display similarity matrix
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = plot_similarity_matrix(sentence_sim, "Sentence Transformers Similarity Matrix")
                st.pyplot(fig)
            
            with col2:
                st.subheader("📈 Similarity Statistics")
                avg_similarity = np.mean(sentence_sim[np.triu_indices_from(sentence_sim, k=1)])
                max_similarity = np.max(sentence_sim[np.triu_indices_from(sentence_sim, k=1)])
                min_similarity = np.min(sentence_sim[np.triu_indices_from(sentence_sim, k=1)])
                
                st.metric("Average Similarity", f"{avg_similarity:.3f}")
                st.metric("Maximum Similarity", f"{max_similarity:.3f}")
                st.metric("Minimum Similarity", f"{min_similarity:.3f}")
            
            # Display clones
            if sentence_clones:
                st.subheader("🚨 Potential Plagiarism Detected")
                for i, (idx1, idx2, similarity) in enumerate(sentence_clones):
                    with st.expander(f"Clone Pair {i+1}: Text {idx1+1} ↔ Text {idx2+1} (Similarity: {similarity:.3f})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text_area(f"Text {idx1+1}", value=texts[idx1], height=100, key=f"clone_text1_{i}")
                        with col2:
                            st.text_area(f"Text {idx2+1}", value=texts[idx2], height=100, key=f"clone_text2_{i}")
            else:
                st.success("✅ No potential plagiarism detected above the threshold.")
        
        # OpenAI results tab
        if use_openai and openai_embeddings and len(result_tabs) > 1:
            with result_tabs[1]:
                st.subheader("🧠 OpenAI Embeddings Results")
                
                # Display similarity matrix
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = plot_similarity_matrix(openai_sim, "OpenAI Embeddings Similarity Matrix")
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("📈 Similarity Statistics")
                    avg_similarity = np.mean(openai_sim[np.triu_indices_from(openai_sim, k=1)])
                    max_similarity = np.max(openai_sim[np.triu_indices_from(openai_sim, k=1)])
                    min_similarity = np.min(openai_sim[np.triu_indices_from(openai_sim, k=1)])
                    
                    st.metric("Average Similarity", f"{avg_similarity:.3f}")
                    st.metric("Maximum Similarity", f"{max_similarity:.3f}")
                    st.metric("Minimum Similarity", f"{min_similarity:.3f}")
                
                # Display clones
                if openai_clones:
                    st.subheader("🚨 Potential Plagiarism Detected")
                    for i, (idx1, idx2, similarity) in enumerate(openai_clones):
                        with st.expander(f"Clone Pair {i+1}: Text {idx1+1} ↔ Text {idx2+1} (Similarity: {similarity:.3f})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.text_area(f"Text {idx1+1}", value=texts[idx1], height=100, key=f"openai_clone_text1_{i}")
                            with col2:
                                st.text_area(f"Text {idx2+1}", value=texts[idx2], height=100, key=f"openai_clone_text2_{i}")
                else:
                    st.success("✅ No potential plagiarism detected above the threshold.")
        
        # Comparison tab
        if use_openai and openai_embeddings and len(result_tabs) > 2:
            with result_tabs[2]:
                st.subheader("🔄 Model Comparison")
                
                # Create comparison dataframe
                comparison_data = []
                for i in range(len(texts)):
                    for j in range(i + 1, len(texts)):
                        comparison_data.append({
                            'Text Pair': f"Text {i+1} ↔ Text {j+1}",
                            'Sentence Transformers': sentence_sim[i][j],
                            'OpenAI Embeddings': openai_sim[i][j],
                            'Difference': abs(sentence_sim[i][j] - openai_sim[i][j])
                        })
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
                # Correlation between models
                st.subheader("📊 Model Correlation")
                correlation = np.corrcoef(
                    sentence_sim[np.triu_indices_from(sentence_sim, k=1)],
                    openai_sim[np.triu_indices_from(openai_sim, k=1)]
                )[0, 1]
                st.metric("Correlation Coefficient", f"{correlation:.3f}")

if __name__ == "__main__":
    main()
