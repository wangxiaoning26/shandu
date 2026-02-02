# Shandu: Advanced Research System Architecture

This directory contains the core architecture of the Shandu deep research system. Our modular design separates concerns and enables future extensibility while maintaining clean, testable code.

## 📊 System Architecture

Shandu implements a sophisticated state-based workflow using LangGraph and LangChain to create a robust, extensible research system:

```
shandu/
├── __init__.py           # Package initialization
├── cli.py                # Command-line interface
├── config.py             # Configuration management
├── prompts.py            # Centralized prompt templates
├── agents/               # Research agent implementations
│   ├── __init__.py
│   ├── agent.py          # LangChain-based agent
│   ├── langgraph_agent.py # LangGraph state-based agent
│   ├── graph/            # Graph workflow components
│   │   ├── __init__.py
│   │   ├── builder.py    # Graph construction
│   │   └── wrapper.py    # Async function wrappers
│   ├── nodes/            # Graph node implementations
│   │   ├── __init__.py
│   │   ├── initialize.py # Research initialization
│   │   ├── reflect.py    # Research reflection
│   │   ├── search.py     # Content search and analysis
│   │   └── ...           # Other node implementations
│   ├── processors/       # Content processing
│   │   ├── __init__.py
│   │   ├── content_processor.py # Content extraction
│   │   └── report_generator.py  # Report generation
│   └── utils/            # Agent utilities
│       ├── __init__.py
│       └── agent_utils.py # Helper functions
├── research/             # Research orchestration
│   ├── __init__.py
│   └── researcher.py     # Result management
├── scraper/              # Web scraping functionality
│   ├── __init__.py
│   └── scraper.py        # Ethical web scraper
└── search/               # Search functionality
    ├── __init__.py
    ├── ai_search.py      # AI-powered search
    └── search.py         # Multi-engine search
```

## 🔄 LangGraph Research Workflow

Shandu's research process follows a sophisticated state-based workflow:

1. **Initialize**: Define research query, parameters, and create a research plan
2. **Reflect**: Analyze current findings and identify knowledge gaps
3. **Generate Queries**: Create targeted search queries based on analysis
4. **Search**: Execute search queries and collect results
5. **Smart Source Selection**: Filter and prioritize the most valuable sources
6. **Format Citations**: Prepare properly formatted citations for all sources
7. **Generate Initial Report**: Create a first draft of the research report
8. **Enhance Report**: Add depth, detail, and proper structure
9. **Expand Key Sections**: Further develop important sections through multi-step synthesis
10. **Finalize Report**: Apply final formatting and quality checks

## 🧠 Advanced Technical Features

### State-Based Research With LangGraph

Our LangGraph implementation provides several key advantages:

- **Clear State Transitions**: Each research phase has well-defined inputs and outputs
- **Conditional Logic**: Dynamically determines next steps based on current state
- **Circular Flow**: Supports recursive exploration until depth conditions are met
- **Parallel Processing**: Handles concurrent operations for efficiency
- **Error Resilience**: Continues functioning even if individual steps encounter issues

### Enhanced Content Processing

Shandu implements sophisticated content processing:

- **Content Relevance Filtering**: Uses AI to determine if content is relevant to the research query
- **Source Reliability Assessment**: Evaluates sources for credibility and authority
- **Main Content Extraction**: Identifies and extracts the primary content from web pages
- **Content Analysis Pipeline**: Multi-step analysis for key information extraction
- **Theme Identification**: Automatically discovers and organizes thematic elements

### Advanced Report Generation

Our multi-step report generation process ensures high-quality output:

1. **Theme Extraction**: Identifies key themes across all research
2. **Initial Report Generation**: Creates a structured first draft
3. **Report Enhancement**: Adds depth, citations, and improved organization
4. **Key Section Expansion**: Further develops the most important sections
5. **Citation Management**: Ensures proper attribution of all sources
6. **Final Cleanup**: Removes artifacts and ensures consistent formatting

## 💻 API Details

### ResearchGraph Class

```python
class ResearchGraph:
    """
    State-based research workflow using LangGraph.
    Provides a structured approach to deep research with multiple stages.
    """
    def __init__(
        self, 
        llm: Optional[ChatOpenAI] = None, 
        searcher: Optional[UnifiedSearcher] = None, 
        scraper: Optional[WebScraper] = None, 
        temperature: float = 0.5,
        date: Optional[str] = None
    )
    
    async def research(
        self, 
        query: str, 
        depth: int = 2, 
        breadth: int = 4, 
        progress_callback: Optional[Callable] = None,
        include_objective: bool = False,
        detail_level: str = "high" 
    ) -> ResearchResult
    
    def research_sync(
        self, 
        query: str, 
        depth: int = 2, 
        breadth: int = 4, 
        progress_callback: Optional[Callable] = None,
        include_objective: bool = False,
        detail_level: str = "high"
    ) -> ResearchResult
```

### AISearcher Class

```python
class AISearcher:
    """
    AI-powered search with content scraping for deeper insights.
    """
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        searcher: Optional[UnifiedSearcher] = None,
        scraper: Optional[WebScraper] = None,
        max_results: int = 10,
        max_pages_to_scrape: int = 3
    )
    
    async def search(
        self, 
        query: str,
        engines: Optional[List[str]] = None,
        detailed: bool = False,
        enable_scraping: bool = True
    ) -> AISearchResult
```

## 🔌 Integration Points

Shandu is designed for easy integration:

- **CLI Interface**: Command-line tools for direct usage
- **Python API**: Clean, well-documented API for integration into other applications
- **Extensible Components**: Easy to add new search engines, scrapers, or processing steps
- **Custom LLM Support**: Works with any LangChain-compatible LLM
- **Callback System**: Progress tracking and event hooks

## 🔍 Implementation Details

### Prompt Engineering

Shandu uses carefully crafted prompts for:

- Query clarification
- Research planning
- Content analysis
- Source evaluation
- Report generation
- Citation formatting

### Async Processing

Extensive use of async/await patterns for:

- Parallel search execution
- Concurrent web scraping
- Efficient content processing
- Responsive UI updates

### Caching System

Multi-level caching for:

- Search results
- Scraped content
- Content analysis
- LLM responses

## 🔬 Research Algorithm

Our research algorithm optimizes for:

1. **Breadth**: Exploring multiple relevant sub-topics
2. **Depth**: Drilling down into important details
3. **Convergence**: Focusing on the most relevant information
4. **Coverage**: Ensuring comprehensive topic exploration
5. **Source Quality**: Prioritizing reliable, authoritative sources
6. **Synthesis**: Creating coherent, well-structured reports

For more information on using Shandu, see the main [README.md](../README.md) file.