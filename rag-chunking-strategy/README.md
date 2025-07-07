# RAG Chunking Strategy Explorer

A comprehensive web application for exploring and visualizing different chunking strategies for Retrieval-Augmented Generation (RAG) systems. Built with Streamlit, this tool helps developers and researchers understand how different chunking approaches affect document segmentation and retrieval performance.

## 🚀 Features

### Core Functionality
- **PDF Upload & Text Extraction**: Upload PDF documents and extract text content
- **Multiple Chunking Strategies**: Explore 7 different chunking approaches
- **Interactive Visualization**: Analyze chunk distributions with interactive charts
- **Strategy Comparison**: Compare different approaches with detailed metrics
- **Export Capabilities**: Download chunk results as CSV files

### Supported Chunking Strategies

1. **Fixed Size Chunking**
   - Splits text into chunks of fixed character size with configurable overlap
   - Best for: General purpose, consistent chunk sizes for embedding models
   - Configurable: Chunk size (200-2000 chars), Overlap (0-500 chars)

2. **Sentence-Based Chunking**
   - Groups sentences together based on sentence boundaries
   - Best for: When semantic coherence at sentence level is important
   - Configurable: Max sentences per chunk (1-10)

3. **Paragraph-Based Chunking**
   - Uses paragraph breaks as natural chunk boundaries
   - Best for: Documents with clear paragraph structure
   - Automatic detection of paragraph boundaries

4. **Semantic Chunking**
   - Uses sentence similarity to group semantically related content
   - Best for: When semantic coherence is critical for retrieval
   - Configurable: Similarity threshold (0.3-0.9)
   - Uses: Sentence-BERT embeddings for similarity calculation

5. **Token-Based Chunking**
   - Splits text based on token count using tiktoken encoding
   - Best for: When working with LLMs that have token limits
   - Configurable: Chunk size (100-1000 tokens), Overlap (0-200 tokens)

6. **Recursive Character Chunking**
   - Recursively splits text using hierarchical separators
   - Best for: Balanced approach for most document types
   - Configurable: Chunk size (200-2000 chars), Overlap (0-500 chars)

7. **Sliding Window Chunking**
   - Creates overlapping chunks using a sliding window approach
   - Best for: When context overlap is crucial for retrieval
   - Configurable: Window size (200-2000 chars), Step size (100-1000 chars)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd rag-chunking-strategy
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - The application will automatically open in your default browser
   - Default URL: `http://localhost:8501`

## 📋 Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pypdf**: PDF text extraction
- **langchain**: Text splitting utilities
- **sentence-transformers**: Semantic similarity calculations
- **tiktoken**: Token counting for OpenAI models

### Visualization & Analysis
- **plotly**: Interactive charts and visualizations
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Additional plotting capabilities

### Natural Language Processing
- **nltk**: Natural language processing utilities
- **transformers**: Hugging Face transformers library

## 📊 Usage Guide

### Basic Usage

1. **Upload PDF Document**
   - Click "Upload PDF Document" in the sidebar
   - Select a PDF file from your computer
   - Wait for text extraction to complete

2. **Choose Chunking Strategy**
   - Select from 7 available strategies in the dropdown
   - Adjust strategy-specific parameters using sliders
   - Parameters update in real-time

3. **Analyze Results**
   - View strategy explanation and use cases
   - Examine chunk statistics and metrics
   - Explore interactive visualizations
   - Browse individual chunks

4. **Export Data**
   - Download chunk results as CSV
   - Use data for further analysis or comparison

### Advanced Features

#### Strategy Comparison
- Upload the same document multiple times
- Try different strategies and parameters
- Compare results using exported CSV files

#### Large Document Handling
- Application handles large PDF files efficiently
- Semantic chunking may take longer for very large documents
- Consider using simpler strategies for initial exploration

#### Parameter Tuning
- **Chunk Size**: Balance between context and granularity
- **Overlap**: Increase for better context preservation
- **Similarity Threshold**: Adjust for semantic coherence
- **Window Size**: Control sliding window chunk size

## 🔍 Understanding Chunking Strategies

### When to Use Each Strategy

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| Fixed Size | General purpose, consistent sizing | Predictable sizes, simple | May break semantic units |
| Sentence-Based | Preserving sentence integrity | Natural boundaries, coherent | Variable sizes |
| Paragraph-Based | Document structure preservation | Maintains structure, coherent blocks | Highly variable sizes |
| Semantic | Semantic coherence critical | Intelligent grouping, coherent | Computationally expensive |
| Token-Based | LLM token limits | Precise token control | May break words/sentences |
| Recursive Character | Balanced approach | Intelligent splitting, preserves structure | Complex logic |
| Sliding Window | High context overlap needed | Context preservation, overlap | More chunks, redundancy |

### Performance Considerations

#### Computational Complexity
- **Fastest**: Fixed Size, Paragraph-Based
- **Moderate**: Sentence-Based, Token-Based, Recursive Character
- **Slowest**: Semantic Chunking (requires embedding computation)

#### Memory Usage
- **Lowest**: Fixed Size, Sentence-Based
- **Moderate**: Most other strategies
- **Highest**: Semantic Chunking (stores embeddings)

#### Chunk Quality vs. Speed Trade-offs
- **High Speed, Lower Quality**: Fixed Size
- **Balanced**: Recursive Character, Sentence-Based
- **High Quality, Lower Speed**: Semantic Chunking

## 🎯 Use Cases

### Research & Development
- **RAG System Development**: Test different chunking strategies for your RAG pipeline
- **Document Analysis**: Understand how different documents respond to various chunking approaches
- **Strategy Comparison**: Compare multiple approaches on the same document

### Educational
- **Learning RAG Concepts**: Understand how chunking affects retrieval performance
- **NLP Education**: Explore different text segmentation approaches
- **Visualization**: See how different strategies segment the same content

### Production Planning
- **Strategy Selection**: Choose the best chunking strategy for your use case
- **Parameter Tuning**: Optimize chunk size and overlap for your specific needs
- **Performance Testing**: Understand computational requirements of different strategies

## 🔧 Technical Architecture

### Application Structure
```
rag-chunking-strategy/
├── app.py                 # Main Streamlit application
├── ui_components.py       # UI components and styling
├── chunking_strategies.py # Chunking logic and strategies
├── utils.py              # Utility functions and helpers
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
└── venv/                 # Virtual environment
```

### Key Components

#### Main Application (`app.py`)
- Orchestrates the entire application flow
- Handles user interactions and state management
- Integrates all modular components

#### UI Components (`ui_components.py`)
- All Streamlit UI elements and styling
- `render_*()` functions for different UI sections
- Custom CSS styling for better UI/UX
- Reusable UI components

#### Chunking Strategies (`chunking_strategies.py`)
- `ChunkingStrategies` class with all 7 strategies
- Consistent interface for all chunking methods
- Token counting and metadata generation
- Strategy information and parameters

#### Utilities (`utils.py`)
- PDF processing and text extraction
- Text cleaning and validation functions
- Helper functions for metrics and analysis
- Error handling and validation

### Data Flow
1. **PDF Upload** → Text Extraction
2. **Strategy Selection** → Parameter Configuration
3. **Chunking Process** → Metadata Generation
4. **Visualization** → Interactive Charts
5. **Export** → CSV Download

## 🚀 Advanced Usage

### Custom Strategy Development
The application is designed to be extensible. To add a new chunking strategy:

1. Add a new method to the `ChunkingStrategies` class
2. Follow the existing pattern for return data structure
3. Add strategy explanation to `display_strategy_explanation()`
4. Update the strategy dropdown in the main function

### Integration with Other Tools
- **Export chunks** as CSV for use in other analysis tools
- **Token counts** compatible with OpenAI API limits
- **Metadata format** suitable for vector database ingestion

### Performance Optimization
- **Caching**: Consider implementing Streamlit caching for large documents
- **Async Processing**: For very large documents, consider async processing
- **Memory Management**: Monitor memory usage with large PDFs

## 🐛 Troubleshooting

### Common Issues

#### PDF Upload Errors
- **Issue**: "Error reading PDF"
- **Solution**: Ensure PDF is not corrupted or password-protected
- **Alternative**: Try a different PDF file

#### Semantic Chunking Slow/Fails
- **Issue**: Semantic chunking takes too long or fails
- **Solution**: Reduce document size or use simpler strategy first
- **Note**: First run downloads the embedding model

#### Memory Issues
- **Issue**: Application crashes with large documents
- **Solution**: Try smaller documents or simpler strategies
- **Tip**: Use Fixed Size or Sentence-Based for large files

#### Missing Dependencies
- **Issue**: Import errors when running the application
- **Solution**: Ensure virtual environment is activated and all dependencies are installed
- **Command**: `pip install -r requirements.txt`

### Performance Tips
1. **Start with smaller documents** to test strategies
2. **Use Fixed Size chunking** for initial exploration
3. **Semantic chunking** works best with well-structured text
4. **Adjust parameters gradually** to see their effects

## 📈 Future Enhancements

### Planned Features
- **Multi-document comparison**: Compare strategies across multiple documents
- **Chunk quality metrics**: Semantic coherence scoring
- **Custom separators**: User-defined splitting rules
- **Batch processing**: Process multiple files at once
- **Advanced visualizations**: Heatmaps, similarity matrices

### Potential Integrations
- **Vector databases**: Direct integration with Pinecone, Weaviate
- **LLM APIs**: Integration with OpenAI, Anthropic APIs
- **Embedding models**: Support for different embedding models
- **File formats**: Support for DOCX, TXT, HTML files

## 🤝 Contributing

We welcome contributions to improve the RAG Chunking Strategy Explorer! Here's how you can help:

### Areas for Contribution
- **New chunking strategies**: Implement additional approaches
- **Performance optimizations**: Improve speed and memory usage
- **UI/UX improvements**: Enhance the user interface
- **Documentation**: Improve guides and examples
- **Bug fixes**: Report and fix issues

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **LangChain**: For excellent text splitting utilities
- **Streamlit**: For the amazing web app framework
- **Sentence Transformers**: For semantic similarity calculations
- **OpenAI**: For tiktoken tokenization
- **Hugging Face**: For transformer models and libraries

---

**Happy Chunking!** 🚀

For questions, issues, or suggestions, please open an issue in the repository or contact the maintainers. 