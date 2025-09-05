# SEC Filings QA System

A question-answering system that analyzes SEC filings to extract insights from financial documents. Built for quantitative research and financial analysis workflows, this system can process multiple companies' filings simultaneously and answer complex queries about revenue, risks, strategies, and financial metrics.

## Features

- **Multi-Company Analysis**: Query across multiple companies simultaneously with automatic ticker detection
- **Semantic Search**: Uses sentence transformers (all-mpnet-base-v2) to find relevant document sections
- **Smart Query Processing**: Automatically detects tickers, years, and document types from natural language queries
- **Source Attribution**: All answers include references to specific filings with relevance scores
- **Numerical Extraction**: Identifies and analyzes financial metrics, revenue figures, and growth rates
- **Document Chunking**: Preserves document structure while creating searchable segments
- **Fallback System**: Demo data ensures functionality even without API access

## Setup

### Dependencies
```bash
pip install transformers torch sentence-transformers faiss-cpu requests beautifulsoup4 pandas numpy tabulate
```

### Quick Start
```python
from sec_analyzer import SECFilingsAnalyzer

# Initialize with SEC API key (get from sec-api.io)
analyzer = SECFilingsAnalyzer(api_key="your_api_key_here")

# Build knowledge base for target companies
analyzer.build_knowledge_base(["AAPL", "MSFT", "GOOGL"], max_docs=10)

# Check system stats
stats = analyzer.get_system_stats()
print(f"Loaded {stats['total_chunks']} chunks from {stats['total_companies']} companies")
```

## Usage

### Basic Queries
```python
# Single company analysis
result = analyzer.answer_question("What are Apple's main revenue sources?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {len(result['sources'])}")
```

### Advanced Queries
```python
# Multi-dimensional queries (ticker + time + document type)
result = analyzer.answer_question("Apple's 2023 10-K risk factors")

# Temporal analysis
result = analyzer.answer_question("How has Microsoft's cloud revenue changed over time?")

# Cross-company comparisons
result = analyzer.compare_companies(
    "How do R&D investments compare?", 
    ["AAPL", "MSFT", "GOOGL"]
)
```

### Query Examples by Category

**Revenue Analysis**
- "What are Apple's primary revenue drivers and growth rates?"
- "Compare revenue composition between Microsoft and Google"
- "How has Services revenue evolved for Apple?"

**Risk Assessment** 
- "What climate-related risks do tech companies identify?"
- "Compare supply chain risks across Apple and Microsoft"
- "What regulatory risks are most commonly cited?"

**Strategic Analysis**
- "How do companies describe their AI and automation strategies?"
- "What M&A activity has Microsoft disclosed recently?"
- "Compare competitive positioning statements"

**Financial Metrics**
- "Compare R&D spending trends across major tech companies"
- "What are the reported margins for cloud services?"
- "Analyze working capital changes and driving factors"

## Architecture

### Document Processing Pipeline
1. **Data Ingestion**: Fetches filings via SEC API or uses demo data
2. **Content Extraction**: Parses HTML documents and cleans text
3. **Chunking Strategy**: Creates ~2500 character segments with 500 character overlaps
4. **Metadata Preservation**: Maintains ticker, filing type, date, and section information

### Query Processing
1. **Intent Recognition**: Extracts tickers, years, and document types using regex and dictionaries
2. **Vector Search**: Encodes query and searches FAISS index for relevant chunks
3. **Context Building**: Assembles relevant document sections with metadata
4. **Answer Generation**: Uses RoBERTa-large-squad2 for question answering

### Technical Stack
- **Embeddings**: sentence-transformers/all-mpnet-base-v2
- **Search**: FAISS IndexFlatIP for cosine similarity
- **QA Models**: deepset/roberta-large-squad2 (primary), roberta-base-squad2 (fallback)
- **Document Processing**: BeautifulSoup for HTML parsing
- **Financial Extraction**: Regex patterns for revenue, R&D, margins, growth rates

## Data Sources

### Supported Filing Types
- **10-K**: Annual reports with comprehensive business overviews
- **10-Q**: Quarterly financial statements and updates  
- **8-K**: Current reports on material events
- **DEF 14A**: Proxy statements with governance and compensation data

### API Integration
- Primary: SEC API (sec-api.io) for real-time filings
- Fallback: Demo data with realistic financial content for testing
- Rate limiting and error handling for robust operation

## Performance Metrics

### Processing Capabilities
- Handles 10-15 companies simultaneously
- Processes documents up to 100KB per filing
- Creates searchable knowledge base in ~2-3 minutes
- Query response time: 1-3 seconds depending on complexity

### Quality Metrics
- Confidence scores for all answers (0.0 to 1.0 scale)
- Source attribution with relevance scoring
- Handles multi-part questions effectively
- Graceful degradation when information is limited

## Configuration

### Supported Companies
Currently configured for major technology companies:
- **Technology**: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, NFLX
- **Financial**: JPM, BAC, WFC, GS, MS, C
- **Healthcare**: JNJ, UNH, PFE, ABBV, MRK
- **Energy**: XOM, CVX, COP, EOG, SLB

### Customization Options
```python
# Adjust chunk size and overlap
analyzer.create_text_chunks(text, chunk_size=3000, overlap=600)

# Filter by specific criteria
results = analyzer.search_documents(
    query, 
    ticker_filter="AAPL",
    form_filter="10-K", 
    date_filter="2023"
)
```

## Example Output

```python
result = analyzer.answer_question("Compare R&D spending between Apple and Microsoft")

# Sample response structure:
{
    'answer': 'Apple reported R&D expenses of $29.9 billion in 2023...',
    'confidence': 0.847,
    'sources': [
        {
            'company': 'Apple Inc. (AAPL)',
            'form_type': '10-K', 
            'filing_date': '2023-10-27',
            'relevance_score': 0.892,
            'chunk_length': 2341
        }
    ],
    'numeric_analysis': {...},
    'query_params': {
        'tickers': ['AAPL', 'MSFT'],
        'years': [],
        'doc_types': []
    }
}
```


## Files Structure

```
sec_analyzer.py          # Main analyzer class with full pipeline
demo_data/              # Synthetic filing content for testing
examples/               # Sample queries and usage patterns
tests/                  # Performance and accuracy tests
requirements.txt        # Dependencies list
```

## Development Notes

Built for systematic analysis of SEC filings in quantitative research workflows. The system prioritizes accuracy and source attribution over speed, making it suitable for research where verifiability is important. The fallback demo system ensures functionality during development and testing phases.
