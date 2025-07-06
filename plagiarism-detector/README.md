# Plagiarism Detector - Streamlit App

A modern web application for detecting plagiarism and semantic similarity between texts using advanced AI models.

## Features

- 🔍 **Semantic Similarity Analysis**: Compare multiple texts using Sentence Transformers and OpenAI embeddings
- 📊 **Interactive Visualizations**: Beautiful heatmaps and similarity matrices
- 📁 **File Upload Support**: Upload text files for comparison
- ⚙️ **Customizable Settings**: Adjust similarity thresholds and preprocessing options
- 🚨 **Plagiarism Detection**: Automatic detection of potential plagiarism above threshold
- 📈 **Detailed Statistics**: Comprehensive similarity metrics and analysis

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Configure your OpenAI API key in `.streamlit/secrets.toml` if you want to use OpenAI embeddings:
```toml
OPENAI_API_KEY = "your-api-key-here"
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Enter your texts manually or upload text files

4. Configure settings in the sidebar:
   - Adjust similarity threshold
   - Enable/disable OpenAI embeddings
   - Toggle text preprocessing

5. Click "Analyze Texts" to compare and detect potential plagiarism

## Models Used

- **Sentence Transformers**: `paraphrase-MiniLM-L6-v2` for fast, local embeddings
- **OpenAI Embeddings**: `text-embedding-ada-002` for high-quality embeddings (requires API key)

## Output

The app provides:
- Similarity matrices with heatmap visualizations
- Detailed statistics (average, maximum, minimum similarity)
- Potential plagiarism pairs above the threshold
- Side-by-side comparison of flagged texts
- Model comparison when using both embedding types

## Requirements

- Python 3.7+
- Internet connection for downloading models
- OpenAI API key (optional, for OpenAI embeddings) 