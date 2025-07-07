"""
RAG Chunking Strategy Explorer
A comprehensive web application for exploring and visualizing different chunking strategies 
for Retrieval-Augmented Generation (RAG) systems.
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import modular components
from ui_components import (
    setup_page_config,
    apply_custom_css,
    render_header,
    render_file_upload,
    render_text_statistics,
    render_strategy_selector,
    render_strategy_parameters,
    display_strategy_explanation,
    render_chunk_statistics,
    render_chunk_visualization,
    render_chunk_details_table,
    render_individual_chunk_viewer,
    render_export_section,
    render_welcome_message,
    show_success_message,
    show_processing_spinner,
    show_error_message
)

from chunking_strategies import (
    ChunkingStrategies,
    get_available_strategies,
    get_strategy_info
)

from utils import (
    extract_text_from_pdf,
    clean_text,
    validate_pdf_file,
    calculate_text_metrics,
    validate_chunk_parameters,
    estimate_processing_time
)

def main():
    """Main application function"""
    # Setup page configuration
    setup_page_config()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Render header
    render_header()
    
    # Initialize chunking strategies
    chunker = ChunkingStrategies()
    
    # Render file upload
    uploaded_file = render_file_upload()
    
    if uploaded_file is not None:
        # Validate PDF file
        is_valid, message = validate_pdf_file(uploaded_file)
        
        if not is_valid:
            show_error_message(message)
            return
        
        # Extract text from PDF
        with show_processing_spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        
        if text:
            # Clean and process text
            text = clean_text(text)
            
            # Show success message
            show_success_message(len(text))
            
            # Calculate text metrics
            metrics = calculate_text_metrics(text)
            token_count = chunker.count_tokens(text)
            
            # Display basic text statistics
            render_text_statistics(text, token_count)
            
            # Strategy selection
            strategy = render_strategy_selector()
            
            # Strategy-specific parameters
            params = render_strategy_parameters(strategy)
            
            # Validate parameters
            param_errors = validate_chunk_parameters(strategy, params)
            if param_errors:
                for error in param_errors:
                    show_error_message(error)
                return
            
            # Estimate processing time
            estimated_time = estimate_processing_time(len(text), strategy)
            
            # Apply chunking strategy
            spinner_message = f"Applying {strategy} chunking strategy..."
            if strategy == 'Semantic':
                spinner_message += " (This may take a while for the first run)"
            
            with show_processing_spinner(spinner_message):
                try:
                    chunks = chunker.apply_strategy(strategy, text, params)
                except Exception as e:
                    show_error_message(f"Error applying {strategy} strategy: {str(e)}")
                    return
            
            # Display strategy explanation
            st.header("Strategy Explanation")
            display_strategy_explanation(strategy)
            
            # Display chunk analysis
            st.header("Chunk Analysis")
            df = render_chunk_statistics(chunks)
            
            if df is not None:
                # Render visualizations
                render_chunk_visualization(chunks)
                
                # Render chunk details table
                render_chunk_details_table(df)
                
                # Render individual chunk viewer
                render_individual_chunk_viewer(chunks)
                
                # Render export section
                render_export_section(df, strategy)
        else:
            show_error_message("Failed to extract text from PDF. Please try a different file.")
    else:
        # Render welcome message and strategy information
        render_welcome_message()

if __name__ == "__main__":
    main() 