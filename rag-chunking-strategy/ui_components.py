import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="RAG Chunking Strategy Explorer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
    """Apply custom CSS styling to the application"""
    st.markdown("""
    <style>
        /* ---------- Common Styles ---------- */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4; /* default (light mode) */
            text-align: center;
            margin-bottom: 2rem;
        }

        .strategy-card,
        .chunk-display,
        .metric-card {
            padding: 1rem;
            border-radius: 10px;
        }

        .strategy-card {
            border-left: 4px solid #1f77b4;
            background-color: #f8f9fa;
            margin: 1rem 0;
        }

        .chunk-display {
            background-color: #e8f4f8;
            border-left: 3px solid #17a2b8;
            margin: 0.5rem 0;
        }

        .metric-card {
            background-color: #f1f3f4;
            text-align: center;
        }

        /* ---------- Dark Mode Overrides ---------- */
        @media (prefers-color-scheme: dark) {
            /* Update header text color for contrast */
            .main-header {
                color: #4dabf7;
            }

            /* Card backgrounds & accent colours */
            .strategy-card {
                background-color: #2f2f35;
                border-left-color: #4dabf7;
            }

            .chunk-display {
                background-color: #3a3a40;
                border-left-color: #66d9e8;
            }

            .metric-card {
                background-color: #2a2a30;
                color: #e1e1e1;
            }

            /* Ensure dataframe/table text remains readable */
            .stDataFrame, .stTable {
                color: #e1e1e1 !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render the main application header"""
    st.markdown('<h1 class="main-header">📄 RAG Chunking Strategy Explorer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Explore different chunking strategies for Retrieval-Augmented Generation (RAG) systems.**
    Upload a PDF document and visualize how different chunking approaches affect content segmentation.
    """)

def render_file_upload():
    """Render the file upload component in sidebar"""
    st.sidebar.header("Configuration")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF Document", 
        type=['pdf'],
        help="Upload a PDF file to analyze different chunking strategies"
    )
    
    return uploaded_file

def render_text_statistics(text, token_count):
    """Render basic text statistics"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Characters", f"{len(text):,}")
    with col2:
        st.metric("Words", f"{len(text.split()):,}")
    with col3:
        st.metric("Tokens (approx)", f"{token_count:,}")
    with col4:
        paragraph_separator = '\n\n'
        st.metric("Paragraphs", f"{len([p for p in text.split(paragraph_separator) if p.strip()]):,}")

def render_strategy_selector():
    """Render the chunking strategy selector"""
    st.sidebar.subheader("Chunking Strategy")
    strategy = st.sidebar.selectbox(
        "Select Chunking Strategy",
        [
            'Fixed Size',
            'Sentence-Based',
            'Paragraph-Based',
            'Semantic',
            'Token-Based',
            'Recursive Character',
            'Sliding Window'
        ]
    )
    
    return strategy

def render_strategy_parameters(strategy):
    """Render strategy-specific parameter controls"""
    params = {}
    
    if strategy == 'Fixed Size':
        params['chunk_size'] = st.sidebar.slider("Chunk Size (characters)", 200, 2000, 1000)
        params['overlap'] = st.sidebar.slider("Overlap (characters)", 0, 500, 200)
    elif strategy == 'Sentence-Based':
        params['max_sentences'] = st.sidebar.slider("Max Sentences per Chunk", 1, 10, 5)
    elif strategy == 'Semantic':
        params['similarity_threshold'] = st.sidebar.slider("Similarity Threshold", 0.3, 0.9, 0.7)
    elif strategy == 'Token-Based':
        params['chunk_size'] = st.sidebar.slider("Chunk Size (tokens)", 100, 1000, 512)
        params['overlap'] = st.sidebar.slider("Overlap (tokens)", 0, 200, 50)
    elif strategy == 'Recursive Character':
        params['chunk_size'] = st.sidebar.slider("Chunk Size (characters)", 200, 2000, 1000)
        params['overlap'] = st.sidebar.slider("Overlap (characters)", 0, 500, 200)
    elif strategy == 'Sliding Window':
        params['window_size'] = st.sidebar.slider("Window Size (characters)", 200, 2000, 1000)
        params['step_size'] = st.sidebar.slider("Step Size (characters)", 100, 1000, 500)
    
    return params

def display_strategy_explanation(strategy):
    """Display explanation for each chunking strategy"""
    explanations = {
        'Fixed Size': {
            'description': 'Splits text into chunks of fixed character size with configurable overlap.',
            'use_cases': 'General purpose, consistent chunk sizes for embedding models',
            'pros': 'Predictable chunk sizes, simple implementation',
            'cons': 'May break sentences or paragraphs unnaturally'
        },
        'Sentence-Based': {
            'description': 'Groups sentences together into chunks based on sentence boundaries.',
            'use_cases': 'When semantic coherence at sentence level is important',
            'pros': 'Preserves sentence integrity, natural language boundaries',
            'cons': 'Variable chunk sizes, may create very small or large chunks'
        },
        'Paragraph-Based': {
            'description': 'Uses paragraph breaks as natural chunk boundaries.',
            'use_cases': 'Documents with clear paragraph structure',
            'pros': 'Maintains document structure, coherent content blocks',
            'cons': 'Highly variable chunk sizes, depends on document formatting'
        },
        'Semantic': {
            'description': 'Uses sentence similarity to group semantically related content.',
            'use_cases': 'When semantic coherence is critical for retrieval',
            'pros': 'Maintains semantic coherence, intelligent content grouping',
            'cons': 'Computationally expensive, requires embedding model'
        },
        'Token-Based': {
            'description': 'Splits text based on token count using tiktoken encoding.',
            'use_cases': 'When working with LLMs that have token limits',
            'pros': 'Precise token control, optimal for LLM processing',
            'cons': 'May break words or sentences at token boundaries'
        },
        'Recursive Character': {
            'description': 'Recursively splits text using hierarchical separators.',
            'use_cases': 'Balanced approach for most document types',
            'pros': 'Intelligent splitting, preserves structure when possible',
            'cons': 'More complex logic, may still break semantic units'
        },
        'Sliding Window': {
            'description': 'Creates overlapping chunks using a sliding window approach.',
            'use_cases': 'When context overlap is crucial for retrieval',
            'pros': 'High overlap ensures context preservation',
            'cons': 'Creates more chunks, potential information redundancy'
        }
    }
    
    if strategy in explanations:
        info = explanations[strategy]
        st.markdown(f"""
        <div class="strategy-card">
            <h4>{strategy} Strategy</h4>
            <p><strong>Description:</strong> {info['description']}</p>
            <p><strong>Use Cases:</strong> {info['use_cases']}</p>
            <p><strong>Pros:</strong> {info['pros']}</p>
            <p><strong>Cons:</strong> {info['cons']}</p>
        </div>
        """, unsafe_allow_html=True)

def render_chunk_statistics(chunks):
    """Render chunk analysis statistics"""
    if not chunks:
        st.error("No chunks were generated. Please try a different strategy or check your document.")
        return None
    
    df = pd.DataFrame(chunks)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chunks", len(chunks))
    with col2:
        st.metric("Avg Characters", f"{df['char_count'].mean():.0f}")
    with col3:
        st.metric("Avg Tokens", f"{df['token_count'].mean():.0f}")
    with col4:
        st.metric("Avg Words", f"{df['word_count'].mean():.0f}")
    
    return df

def create_chunk_visualization(chunk_data):
    """Create visualizations for chunk analysis"""
    df = pd.DataFrame(chunk_data)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Character Count Distribution', 'Token Count Distribution',
                       'Word Count Distribution', 'Chunk Size Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Character count histogram
    fig.add_trace(
        go.Histogram(x=df['char_count'], name='Character Count', nbinsx=20),
        row=1, col=1
    )
    
    # Token count histogram
    fig.add_trace(
        go.Histogram(x=df['token_count'], name='Token Count', nbinsx=20),
        row=1, col=2
    )
    
    # Word count histogram
    fig.add_trace(
        go.Histogram(x=df['word_count'], name='Word Count', nbinsx=20),
        row=2, col=1
    )
    
    # Chunk size comparison
    fig.add_trace(
        go.Scatter(x=df['chunk_id'], y=df['char_count'], 
                  mode='lines+markers', name='Characters'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    return fig

def render_chunk_visualization(chunks):
    """Render the chunk distribution visualization"""
    st.subheader("Chunk Distribution Visualization")
    fig = create_chunk_visualization(chunks)
    st.plotly_chart(fig, use_container_width=True)

def render_chunk_details_table(df):
    """Render the chunk details table"""
    st.subheader("Chunk Details")
    st.dataframe(
        df[['chunk_id', 'char_count', 'token_count', 'word_count']],
        use_container_width=True
    )

def render_individual_chunk_viewer(chunks):
    """Render individual chunk viewer"""
    st.subheader("Individual Chunks")
    chunk_to_display = st.selectbox(
        "Select chunk to view",
        range(len(chunks)),
        format_func=lambda x: f"Chunk {x} ({chunks[x]['char_count']} chars, {chunks[x]['token_count']} tokens)"
    )
    
    if chunk_to_display is not None:
        chunk = chunks[chunk_to_display]
        st.markdown(f"""
        <div class="chunk-display">
            <h5>Chunk {chunk['chunk_id']}</h5>
            <p><strong>Characters:</strong> {chunk['char_count']} | 
               <strong>Tokens:</strong> {chunk['token_count']} | 
               <strong>Words:</strong> {chunk['word_count']}</p>
            <hr>
            <p>{chunk['content']}</p>
        </div>
        """, unsafe_allow_html=True)

def render_export_section(df, strategy):
    """Render the export section"""
    st.subheader("Export Results")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Chunks as CSV",
        data=csv,
        file_name=f"{strategy.lower().replace(' ', '_')}_chunks.csv",
        mime="text/csv"
    )

def render_welcome_message():
    """Render welcome message when no file is uploaded"""
    st.info("👆 Please upload a PDF file to begin chunking analysis")
    
    # Show example strategies without file
    st.header("Available Chunking Strategies")
    for strategy in ['Fixed Size', 'Sentence-Based', 'Paragraph-Based', 'Semantic', 
                    'Token-Based', 'Recursive Character', 'Sliding Window']:
        display_strategy_explanation(strategy)

def show_success_message(text_length):
    """Show success message after PDF processing"""
    st.success(f"Successfully extracted {text_length} characters from PDF")

def show_processing_spinner(message):
    """Show processing spinner with message"""
    return st.spinner(message)

def show_error_message(message):
    """Show error message"""
    st.error(message) 