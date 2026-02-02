"""
Content processing utilities for research agents.
Contains functionality for handling search results, extracting content, and analyzing information.
"""
from typing import List, Dict, Optional, Any, Union, TypedDict, Sequence
from dataclasses import dataclass
import json
import time
import asyncio
import re
from datetime import datetime
from rich.console import Console
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from ...search.search import SearchResult
from ...scraper import WebScraper, ScrapedContent

console = Console()

class AgentState(TypedDict):
    messages: Sequence[Union[HumanMessage, AIMessage]]
    query: str
    depth: int
    breadth: int
    current_depth: int
    findings: str
    sources: List[Dict[str, Any]]
    selected_sources: List[str]
    formatted_citations: str
    subqueries: List[str]
    content_analysis: List[Dict[str, Any]]
    start_time: float
    chain_of_thought: List[str]
    status: str
    current_date: str
    detail_level: str
    identified_themes: str
    initial_report: str
    enhanced_report: str
    final_report: str

async def is_relevant_url(llm: ChatOpenAI, url: str, title: str, snippet: str, query: str) -> bool:
    """
    Check if a URL is relevant to the query using heuristics and LLM.
    
    Args:
        llm: The language model to use
        url: URL to evaluate
        title: Title of the page
        snippet: Snippet or description of the page
        query: Original query
        
    Returns:
        True if relevant, False otherwise
    """
    # First use simple heuristics to avoid LLM calls for obviously irrelevant domains
    irrelevant_domains = ["pinterest", "instagram", "facebook", "twitter", "youtube", "tiktok",
                         "reddit", "quora", "linkedin", "amazon.com", "ebay.com", "etsy.com",
                         "walmart.com", "target.com"]
    if any(domain in url.lower() for domain in irrelevant_domains):
        return False
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are evaluating search results for relevance to a specific query.
        
        DETERMINE if the search result is RELEVANT or NOT RELEVANT to answering the query.
        
        Output ONLY "RELEVANT" or "NOT RELEVANT" based on your analysis.
        """),
        ("user", """Query: {query}
        
        Search Result:
        Title: {title}
        URL: {url}
        Snippet: {snippet}
        
        Is this result relevant to the query? Answer with ONLY "RELEVANT" or "NOT RELEVANT".
        """)
    ])
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({"query": query, "title": title, "url": url, "snippet": snippet})
    return "RELEVANT" in result.upper()

async def process_scraped_item(llm: ChatOpenAI, item: ScrapedContent, subquery: str, main_content: str) -> Dict[str, Any]:
    """
    Process a scraped item to evaluate reliability and extract content.
    
    Args:
        llm: The language model to use
        item: The scraped content
        subquery: The query used to find this content
        main_content: The main extracted content
        
    Returns:
        Dictionary with processed results
    """
    # Combined reliability evaluation and content extraction
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are analyzing web content for reliability and extracting the most relevant information.
        
        First, evaluate the RELIABILITY of the content using these criteria:
        1. Source credibility and expertise
        2. Evidence quality
        3. Consistency with known facts
        4. Publication date recency
        5. Presence of citations or references
        
        Rate the source as "HIGH", "MEDIUM", or "LOW" reliability with a brief explanation.
        
        Then, EXTRACT the most relevant and valuable content related to the query.
        
        Format your response as:
        RELIABILITY: HIGH/MEDIUM/LOW (brief justification)
        
        EXTRACTED_CONTENT: 
        [Extracted content here - organized, focused on key facts and details]
        """),
        ("user", """Analyze this web content:
        
        URL: {url}
        Title: {title}
        Query: {query}
        
        Content:
        {content}
        """)
    ])
    
    chain = prompt | llm
    result = await chain.ainvoke({
        "url": item.url, 
        "title": item.title, 
        "query": subquery,
        "content": main_content[:8000]
    })
    
    # Parse the combined response
    response_text = result.content
    reliability_section = ""
    content_section = ""
    
    if "RELIABILITY:" in response_text and "EXTRACTED_CONTENT:" in response_text:
        parts = response_text.split("EXTRACTED_CONTENT:")
        reliability_section = parts[0].replace("RELIABILITY:", "").strip()
        content_section = parts[1].strip()
    else:
        # Fallback if format wasn't followed
        reliability_section = "MEDIUM (Unable to parse reliability assessment)"
        content_section = response_text
    
    # Extract rating
    rating = "MEDIUM"
    if "HIGH" in reliability_section.upper():
        rating = "HIGH"
    elif "LOW" in reliability_section.upper():
        rating = "LOW"
    
    justification = reliability_section.replace("HIGH", "").replace("MEDIUM", "").replace("LOW", "").strip()
    if justification.startswith("(") and justification.endswith(")"):
        justification = justification[1:-1].strip()
    
    return {
        "item": item,
        "rating": rating,
        "justification": justification,
        "content": content_section
    }

async def analyze_content(llm: ChatOpenAI, subquery: str, content_text: str) -> str:
    """
    Analyze content from multiple sources and synthesize the information.
    
    Args:
        llm: The language model to use
        subquery: The query that led to this content
        content_text: Combined text from multiple sources
        
    Returns:
        Analysis of the content
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are analyzing and synthesizing information from multiple web sources.
        
        Your task is to:
        1. Identify the most important and relevant information related to the query
        2. Organize the information into a coherent analysis
        3. Highlight key findings, points of consensus, and any contradictions
        4. Maintain source attributions when presenting facts or claims
        
        Create a thorough, well-structured analysis that captures the most valuable insights.
        """),
        ("user", """Analyze the following content related to the query: "{query}"
        
        {content}
        
        Provide a comprehensive analysis that synthesizes the most relevant information
        from these sources, organized into a well-structured format with key findings.
        """)
    ])
    
    # Use more tokens but with a timeout to avoid hanging
    analysis_llm = llm.with_config({"max_tokens": 8192, "timeout": 180})
    analysis_chain = prompt | analysis_llm
    
    try:
        analysis = analysis_chain.invoke({"query": subquery, "content": content_text})
        
        # Clean up any potential artifacts in the analysis
        analysis_content = re.sub(r'Completed:.*?\n', '', analysis.content)
        analysis_content = re.sub(r'Here are.*?search queries.*?\n', '', analysis_content)
        analysis_content = re.sub(r'\*Generated on:.*?\*', '', analysis_content)
        
        # Replace the original content with cleaned content
        return analysis_content
    except Exception as e:
        console.print(f"[dim red]Error in content analysis for {subquery}: {str(e)}. Using simpler analysis.[/dim red]")
        # Fallback to simpler analysis with fewer tokens
        simple_analysis_llm = llm.with_config({"max_tokens": 2048, "timeout": 60})
        simple_analysis_chain = prompt | simple_analysis_llm
        analysis = simple_analysis_chain.invoke({"query": subquery, "content": content_text[:5000]})
        return analysis.content