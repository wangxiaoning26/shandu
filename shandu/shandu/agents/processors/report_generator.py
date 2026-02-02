"""Report generation utilities."""
from typing import List, Dict, Optional, Any, Union
import re
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

async def generate_title(llm: ChatOpenAI, query: str) -> str:
    """Generate a professional title for the report."""
    # First, extract any obvious topic from the query
    topic = query
    
    # More aggressively remove "Based on our discussion" format
    if "Based on our discussion" in query:
        # Extract only the core subject matter without any framework text
        match = re.search(r"(?:.*?(?:refined topic:|research the following:|research|exploring|analyze)):?\s*(.*?)(?:\.|\n|$)", query, re.DOTALL | re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
    
    # Check for other common patterns in the query
    elif ":" in query and len(query.split(":")[0].split()) < 5:
        # Extract the part after the colon if the first part is short (likely a prefix)
        topic = query.split(":", 1)[1].strip()
    
    # Create a title generation prompt with strict guidelines
    title_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are creating a professional, concise title for a research report.
        
        CRITICAL REQUIREMENTS - YOUR TITLE MUST:
        1. Be EXTREMELY CONCISE (8 words maximum)
        2. Be DESCRIPTIVE of the main topic
        3. Be PROFESSIONAL in tone
        4. NEVER start with words like "Evaluating", "Analyzing", "Assessment", "Study", "Investigation", etc.
        5. NEVER contain generic phrases like "A Comprehensive Overview", "An In-depth Look", etc.
        6. NEVER use the word "report", "research", "analysis", or similar meta-terms
        7. NEVER include prefixes like "Topic:", "Subject:", etc.
        8. NEVER be in question format
        9. NEVER be in sentence format - should be a noun phrase
        
        EXAMPLES OF GOOD TITLES:
        "NVIDIA RTX 5000 Series Gaming Performance"
        "Quantum Computing Applications in Cryptography"
        "Clean Energy Transition: Global Market Trends"
        
        CREATE ONLY THE TITLE - NO EXPLANATIONS, PREAMBLE OR COMMENTARY."""),
        ("user", f"Create a professional, concise title (8 words max) for research about: {topic}")
    ])
    
    # Use a more deterministic approach with minimal temperature
    deterministic_llm = llm.with_config({"temperature": 0.1})
    title_chain = title_prompt | deterministic_llm 
    generated_title = (await title_chain.ainvoke({})).content
    
    # Clean up the title
    clean_title = generated_title.strip('"\'').strip()
    clean_title = re.sub(r'^(Title:|Topic:|Subject:|Research Report:|Report:)\s*', '', clean_title, flags=re.IGNORECASE)
    
    # Further processing to make absolutely sure the title is clean
    if ":" in clean_title and len(clean_title.split(":")[0].split()) < 3:
        # Keep only what's after the colon if the first part is short
        clean_title = clean_title.split(":", 1)[1].strip()
        
    # Additional failsafes to enforce guidelines
    if any(clean_title.lower().startswith(word) for word in ["evaluating", "analyzing", "assessment", "study", "investigation", "a comprehensive", "an in-depth"]):
        # Extract the key subject if title starts with unwanted words
        words = clean_title.split()
        if len(words) > 3:
            clean_title = " ".join(words[1:])  # Skip the first word
            
    # Create a shorter, concise title if still too long (more than 8 words)
    if len(clean_title.split()) > 8:
        clean_title = " ".join(clean_title.split()[:8])
        
    # Make sure it's properly capitalized
    words = clean_title.split()
    if words:
        capitalized_words = []
        for word in words:
            if word.lower() in ['a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'in', 'of']:
                capitalized_words.append(word.lower())
            else:
                capitalized_words.append(word.capitalize())
        clean_title = " ".join(capitalized_words)
        
    # Capitalize first word always
    if words:
        words[0] = words[0].capitalize()
        clean_title = " ".join(words)
        
    return clean_title

async def format_citations(llm: ChatOpenAI, selected_sources: List[str], sources: List[Dict[str, Any]]) -> str:
    """Format citations for selected sources."""
    if not selected_sources:
        return ""
    
    # Format the sources for citation
    sources_text = ""
    for i, url in enumerate(selected_sources, 1):
        # Find the source metadata if available
        source_meta = {}
        for source in sources:
            if source.get("url") == url:
                source_meta = source
                break
        
        # Add source information to the text
        sources_text += f"Source {i}:\nURL: {url}\n"
        if source_meta.get("title"):
            sources_text += f"Title: {source_meta.get('title')}\n"
        if source_meta.get("source"):
            sources_text += f"Publication: {source_meta.get('source')}\n"
        if source_meta.get("date"):
            sources_text += f"Date: {source_meta.get('date')}\n"
        sources_text += "\n"
    
    # Use an enhanced citation formatter prompt with stricter formatting requirements
    citation_prompt = """Format the following source information into properly numbered citations for a research report.
    
    FORMATTING REQUIREMENTS:
    1. Each citation MUST start with [n] where n is the citation number
    2. Each citation MUST include the following elements (when available):
       - Publication name or website name in italics
       - Author(s) with full names when available
       - Title of the article/page in quotes
       - Publication date in format: YYYY-MM-DD
       - URL
    
    EXAMPLE PROPER CITATIONS:
    [1] *TechReview*, John Smith, "Advances in GPU Architecture", 2024-01-15, https://techreview.com/articles/gpu-advances
    [2] *ArXiv*, Zhang et al., "Neural Network Performance Optimization", 2023-11-30, https://arxiv.org/papers/nn-optimization
    
    MISSING INFORMATION:
    - If publication name is missing, use the domain name from the URL
    - If author is missing, omit this element
    - If date is missing, use "n.d." (no date)
    
    FORMAT ALL CITATIONS IN A CONSISTENT STYLE.
    Number citations sequentially starting from [1].
    Place each citation on a new line.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", citation_prompt),
        ("user", f"Format these sources into proper citations:\n\n{sources_text}")
    ])
    
    formatter_chain = prompt | llm
    formatted_citations = (await formatter_chain.ainvoke({})).content
    
    # Verify citations have proper format [n] at the beginning
    if not re.search(r'\[\d+\]', formatted_citations):
        # Add fallback formatting if needed
        citations = []
        for i, url in enumerate(selected_sources, 1):
            source_meta = next((s for s in sources if s.get("url") == url), {})
            title = source_meta.get("title", "Untitled")
            source = source_meta.get("source", url.split("//")[1].split("/")[0] if "//" in url else "Unknown Source")
            date = source_meta.get("date", "n.d.")
            
            citation = f"[{i}] *{source}*, \"{title}\", {date}, {url}"
            citations.append(citation)
        
        formatted_citations = "\n".join(citations)
    
    return formatted_citations

async def extract_themes(llm: ChatOpenAI, findings: str) -> str:
    """Extract key themes from research findings."""
    theme_extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are analyzing research findings to identify key themes for a report structure.
        Extract 4-7 major themes from the content that would make logical report sections.
        These themes should emerge naturally from the content rather than following a predetermined structure.
        For each theme, provide a brief description of what content would be included.
        Format your response as a simple list of themes without numbering or additional commentary."""),
        ("user", f"Analyze these research findings and extract 4-7 key themes that should be used as main sections in a report:\n\n{findings}")
    ])
    
    theme_chain = theme_extraction_prompt | llm 
    extracted_themes = (await theme_chain.ainvoke({})).content
    
    return extracted_themes

async def generate_initial_report(
    llm: ChatOpenAI,
    query: str,
    findings: str,
    extracted_themes: str,
    report_title: str,
    selected_sources: List[str],
    formatted_citations: str,
    current_date: str,
    detail_level: str,
    include_objective: bool
) -> str:
    """Generate the initial report draft."""
    # Add objective instruction based on include_objective flag
    objective_instruction = ""
    if not include_objective:
        objective_instruction = "\n\nIMPORTANT: DO NOT include an \"Objective\" section at the beginning of the report. Let your content and analysis naturally determine the structure."
        
    # Use the centralized report generation prompt with dynamic structure guidance
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are generating a comprehensive research report based on extensive research.
        
        REPORT REQUIREMENTS:
        - The report should be thorough, detailed, and professionally formatted in Markdown.
        - Include headers, subheaders, and formatting for readability.
        - The level of detail should be {detail_level.upper()}.
        - Base the report ENTIRELY on the provided research findings.
        - As of {current_date}, incorporate the most up-to-date information available.
        - Include proper citations to sources throughout the text using [n] format.
        - Create a dynamic structure based on the content themes rather than a rigid template.{objective_instruction}
        """),
        ("user", f"""Create an extensive, in-depth research report on this topic.

Title: {report_title}
Analyzed Findings: {findings}
Number of sources: {len(selected_sources)}
Key themes identified in the research: 
{extracted_themes}

Organize your report around these key themes that naturally emerged from the research.
Create a dynamic, organic structure that best presents the findings, rather than forcing content into predetermined sections.
Ensure comprehensive coverage while maintaining a logical flow between topics.

Your report must be extensive, detailed, and grounded in the research. Include all relevant data, examples, and insights found in the research.
Use proper citations to the sources throughout.

IMPORTANT: Begin your report with the exact title provided: "{report_title}" - do not modify or rephrase it.""")
    ])
    
    # Use a significantly higher token limit for comprehensive report generation
    report_llm = llm.with_config({"max_tokens": 16384})
    
    # Format the selected sources for the report
    sources_text = "\n\nSOURCES ANALYZED IN DETAIL:\n"
    if formatted_citations:
        sources_text += formatted_citations
    else:
        for i, url in enumerate(selected_sources, 1):
            sources_text += f"{i}. {url}\n"
    
    # Augment findings with selected source information
    augmented_findings = findings + sources_text
    
    # Generate the initial detailed report
    report_chain = prompt | report_llm
    initial_report = (await report_chain.ainvoke({
        "query": query,
        "analyzed_findings": augmented_findings,
        "num_sources": len(selected_sources)
    })).content
    
    # Apply comprehensive cleanup of artifacts and unwanted sections
    initial_report = re.sub(r'Completed:.*?\n', '', initial_report)
    initial_report = re.sub(r'Here are.*?(search queries|queries to investigate).*?\n', '', initial_report)
    initial_report = re.sub(r'Generated search queries:.*?\n', '', initial_report)
    initial_report = re.sub(r'\*Generated on:.*?\*', '', initial_report)
    
    # Remove entire Research Framework sections (from start to first actual content section)
    if "Research Framework:" in initial_report or "# Research Framework:" in initial_report:
        # Find the start of Research Framework section
        framework_matches = re.search(r'(?:^|\n)(?:#\s*)?Research Framework:.*?(?=\n#|\n\*\*|\Z)', initial_report, re.DOTALL)
        if framework_matches:
            framework_section = framework_matches.group(0)
            initial_report = initial_report.replace(framework_section, '')
    
    # More aggressive removal of "Based on our discussion" text anywhere in the report
    initial_report = re.sub(r'(?:#\s*)?Based on our discussion,.*?(?=\n)', '', initial_report, flags=re.MULTILINE)
    initial_report = re.sub(r'\n\s*?Based on our discussion,.*?(?=\n)', '\n', initial_report, flags=re.MULTILINE)
    initial_report = re.sub(r'Based on our discussion.*?(?=\.)\.', '', initial_report, flags=re.IGNORECASE)
    
    # Try to catch Objective sections and other framework components
    for section in ['Objective', 'Key Aspects to Focus On', 'Constraints and Preferences', 
                   'Areas to Explore in Depth', 'Preferred Sources, Perspectives, or Approaches',
                   'Scope, Boundaries, and Context']:
        initial_report = re.sub(f'^{section}:.*?\n\n', '', initial_report, flags=re.MULTILINE | re.DOTALL)
    
    # Also remove any remaining individual problem framework lines
    for line in ['Research Framework', 'Key Findings', 'Key aspects to focus on']:
        initial_report = re.sub(f'^{line}:.*?\n', '', initial_report, flags=re.MULTILINE)
    
    # Remove any lingering meta-commentary about research process
    for pattern in [
        r'\n\s*?The following research explores.*?(?=\n)', 
        r'\n\s*?This research report aims to.*?(?=\n)',
        r'\n\s*?This report examines.*?(?=\n)', 
        r'\n\s*?I\'ll research the following.*?(?=\n)'
    ]:
        initial_report = re.sub(pattern, '\n', initial_report, flags=re.MULTILINE | re.IGNORECASE)
    
    # Ensure the report has the correctly formatted title at the beginning
    # Check if the report already starts with a title (# Something)
    title_match = re.match(r'^#\s+.*?\n', initial_report)
    if title_match:
        # Replace existing title with our generated one
        initial_report = re.sub(r'^#\s+.*?\n', f'# {report_title}\n', initial_report, count=1)
    else:
        # Add our title at the beginning
        initial_report = f'# {report_title}\n\n{initial_report}'
    
    return initial_report

async def enhance_report(llm: ChatOpenAI, initial_report: str, current_date: str, formatted_citations: str, selected_sources: List[Dict], sources: List[Dict]) -> str:
    """Enhance the report with additional detail."""
    # Use the report enhancement prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are enhancing a research report with additional depth, detail and clarity.
        
        Your task is to:
        1. Add more detailed explanations to key concepts
        2. Expand on examples and case studies
        3. Enhance the analysis and interpretation of findings
        4. Improve the overall structure and flow
        5. Add relevant statistics, data points, or evidence from the sources
        6. Ensure proper citation [n] format throughout
        7. Maintain scientific accuracy and up-to-date information (current as of {current_date})
        
        DO NOT add information not supported by the research. Focus on enhancing what's already there.
        """),
        ("user", f"""Enhance this research report with additional depth and detail:

{initial_report}

Make it more comprehensive, rigorous, and valuable to readers while maintaining scientific accuracy.
""")
    ])
    
    # Use high token limit for enhancement
    report_llm = llm.with_config({"max_tokens": 16384})
    
    # Enhance the report
    enhance_chain = prompt | report_llm
    enhanced_report = (await enhance_chain.ainvoke({})).content
    
    # Apply comprehensive cleanup of artifacts and unwanted sections
    final_report = re.sub(r'Completed:.*?\n', '', enhanced_report)
    final_report = re.sub(r'Here are.*?(search queries|queries to investigate).*?\n', '', final_report)
    final_report = re.sub(r'Generated search queries:.*?\n', '', final_report)
    final_report = re.sub(r'\*Generated on:.*?\*', '', final_report)
    
    # Remove entire Research Framework sections (from start to first actual content section)
    if "Research Framework:" in final_report or "# Research Framework:" in final_report:
        # Find the start of Research Framework section
        framework_matches = re.search(r'(?:^|\n)(?:#\s*)?Research Framework:.*?(?=\n#|\n\*\*|\Z)', final_report, re.DOTALL)
        if framework_matches:
            framework_section = framework_matches.group(0)
            final_report = final_report.replace(framework_section, '')
    
    # More aggressive removal of "Based on our discussion" text anywhere in the report
    final_report = re.sub(r'(?:#\s*)?Based on our discussion,.*?(?=\n)', '', final_report, flags=re.MULTILINE)
    final_report = re.sub(r'\n\s*?Based on our discussion,.*?(?=\n)', '\n', final_report, flags=re.MULTILINE)
    final_report = re.sub(r'Based on our discussion.*?(?=\.)\.', '', final_report, flags=re.IGNORECASE)
    
    # Try to catch Objective sections and other framework components
    for section in ['Objective', 'Key Aspects to Focus On', 'Constraints and Preferences', 
                   'Areas to Explore in Depth', 'Preferred Sources, Perspectives, or Approaches',
                   'Scope, Boundaries, and Context']:
        final_report = re.sub(f'^{section}:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    
    # Also remove any remaining individual problem framework lines
    for line in ['Research Framework', 'Key Findings', 'Key aspects to focus on']:
        final_report = re.sub(f'^{line}:.*?\n', '', final_report, flags=re.MULTILINE)
    
    # Remove any sections with headings like "Search Queries" or similar
    final_report = re.sub(r'#+\s+(?:Search Queries|Generated Queries|Queries to Investigate|Research Methodology|Methodology).*?\n(?=#+|\Z)', '', final_report, flags=re.DOTALL)
    
    # Remove any lingering meta-commentary about research process
    for pattern in [
        r'\n\s*?The following research explores.*?(?=\n)', 
        r'\n\s*?This research report aims to.*?(?=\n)',
        r'\n\s*?This report examines.*?(?=\n)', 
        r'\n\s*?I\'ll research the following.*?(?=\n)'
    ]:
        final_report = re.sub(pattern, '\n', final_report, flags=re.MULTILINE | re.IGNORECASE)
    
    # Extract title from initial report to maintain consistency
    title_match = re.match(r'^#\s+(.*?)\n', initial_report)
    report_title = title_match.group(1) if title_match else "Research Report"
    
    # Check if the report already starts with a title (# Something)
    title_match = re.match(r'^#\s+.*?\n', final_report)
    if title_match:
        # Replace existing title with our generated one
        final_report = re.sub(r'^#\s+.*?\n', f'# {report_title}\n', final_report, count=1)
    else:
        # Add our title at the beginning
        final_report = f'# {report_title}\n\n{final_report}'
        
    # Check for malformed references and fix them if needed
    if "References" in final_report:
        # Extract references section
        references_match = re.search(r'#+\s*References.*?(?=#+\s+|\Z)', final_report, re.DOTALL)
        if references_match:
            references_section = references_match.group(0)
            
            # Check if references are properly formatted
            if not re.search(r'\[\d+\]', references_section):
                # Replace problematic references section with properly formatted one
                if formatted_citations:
                    new_references = f"# References\n\n{formatted_citations}\n"
                    final_report = final_report.replace(references_section, new_references)
                else:
                    # Generate basic references if formatted_citations isn't available
                    basic_references = []
                    for i, url in enumerate(selected_sources, 1):
                        source_meta = next((s for s in sources if s.get("url") == url), {})
                        title = source_meta.get("title", "Untitled")
                        source = source_meta.get("source", url.split("//")[1].split("/")[0] if "//" in url else "Unknown Source")
                        date = source_meta.get("date", "n.d.")
                        
                        citation = f"[{i}] *{source}*, \"{title}\", {date}, {url}"
                        basic_references.append(citation)
                    
                    new_references = f"# References\n\n" + "\n".join(basic_references) + "\n"
                    final_report = final_report.replace(references_section, new_references)
    
    return final_report

async def expand_key_sections(
    llm: ChatOpenAI, 
    report: str, 
    identified_themes: str, 
    current_date: str
) -> str:
    """Expand key sections of the report."""
    # Make sure we have a properly formatted report to start with
    if not report or len(report.strip()) < 1000:
        return report
    
    # Use a more sophisticated approach to identify the most important content areas to expand
    section_analyzer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are analyzing a research report to identify the most critical sections that would benefit from expansion.
        Look for sections that cover core aspects of the research topic but could use more depth, examples, or technical detail.
        Consider which sections would add the most value to the reader if expanded.
        Ignore introduction, conclusion, and references sections.
        Focus on substantive content sections where additional depth would significantly enhance the report.
        Return a list of 3-4 section titles and a brief explanation of why each should be expanded."""),
        ("user", f"Analyze this research report and identify 3-4 sections that would most benefit from expansion:\n\n{report}")
    ])
    
    analyze_chain = section_analyzer_prompt | llm
    analysis_result = (await analyze_chain.ainvoke({})).content
    
    # Extract section titles from the analysis
    section_titles = re.findall(r'(?:"([^"]+)"|\'([^\']+)\'|(?:^|\n)- ([^\n]+))', analysis_result)
    section_titles = [next(title for title in titles if title) for titles in section_titles]
    
    if not section_titles and identified_themes:
        # Fallback to using the previously identified themes
        section_titles = identified_themes.split("\n")[:4]
    
    # Extract each section and its content from the report
    sections_to_expand = []
    for title in section_titles:
        # Try to find exact section title
        pattern = re.compile(rf'(#+\s+{re.escape(title)}\s*\n(?:(?!^#+ ).+\n?)+)', re.MULTILINE)
        matches = pattern.findall(report)
        
        if matches:
            sections_to_expand.append(matches[0])
        else:
            # Try looser matching if exact match fails
            title_words = title.split()
            for i in range(len(title_words), 0, -1):
                partial_title = ' '.join(title_words[:i])
                pattern = re.compile(rf'(#+\s+[^\n]*{re.escape(partial_title)}[^\n]*\s*\n(?:(?!^#+ ).+\n?)+)', re.MULTILINE)
                matches = pattern.findall(report)
                if matches:
                    sections_to_expand.append(matches[0])
                    break
    
    # If we still have no sections, fall back to pattern-based extraction
    if not sections_to_expand:
        section_pattern = r'(#+\s+[^\n]+\n+(?:(?!#+\s+)[^\n]*\n+)*)'
        all_sections = re.findall(section_pattern, report)
        
        # Filter out introduction, conclusion, references, etc.
        content_sections = [
            s for s in all_sections 
            if len(s) > 500 and not any(
                term in s[:50].lower() for term in 
                ["introduction", "conclusion", "reference", "methodology", "executive summary"]
            )
        ]
        
        # Take the 3 longest content sections
        sections_to_expand = sorted(content_sections, key=len, reverse=True)[:3]
    
    # Use multi-step synthesis to expand each section in a more sophisticated way
    expanded_report = report
    total_sections = len(sections_to_expand)
    
    for i, section in enumerate(sections_to_expand, 1):
        # Extract the section title for logging
        section_title = re.match(r'#+\s+([^\n]+)', section)
        section_title = section_title.group(1) if section_title else "Unnamed section"
        
        # Use multi-step synthesis for deeper expansion
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are expanding a key section of a research report with additional depth and detail.
            
            EXPANSION REQUIREMENTS:
            1. Triple the length and detail of the section while maintaining accuracy
            2. Add specific examples, case studies, or data points to support claims
            3. Include additional context and background information
            4. Add nuance, caveats, and alternative perspectives
            5. Use proper citation format [n] throughout
            6. Maintain the existing section structure but add subsections if appropriate
            7. Ensure all information is accurate as of {current_date}
            
            This is step {i} of {total_sections} in expanding the report section by section.
            """),
            ("user", f"""Expand this section with much greater depth and detail:

{section}

Make it substantially more comprehensive while maintaining accuracy and relevance.
Keep the original section heading but expand everything underneath it.
""")
        ])
        
        expansion_chain = prompt | llm
        expanded_section = (await expansion_chain.ainvoke({})).content
        
        # Replace the original section with the expanded one
        expanded_report = expanded_report.replace(section, expanded_section)
    
    return expanded_report