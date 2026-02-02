"""Report generation nodes."""
import re
import time
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from langchain_core.messages import AIMessage
from ..processors.content_processor import AgentState
from ..processors.report_generator import (
    generate_title, 
    extract_themes, 
    generate_initial_report,
    enhance_report,
    expand_key_sections
)
from ..utils.agent_utils import log_chain_of_thought, _call_progress_callback

console = Console()

async def generate_initial_report_node(llm, include_objective, progress_callback, state: AgentState) -> AgentState:
    """Generate the initial report."""
    state["status"] = "Generating initial report"
    console.print("[bold blue]Generating initial comprehensive report with dynamic structure...[/]")

    current_date = state["current_date"]
    
    # Generate a professional title for the report
    report_title = await generate_title(llm, state['query'])
    console.print(f"[bold green]Generated title: {report_title}[/]")
    
    # Extract themes using our processor function
    extracted_themes = await extract_themes(llm, state['findings'])
    
    # Generate the initial report using our processor functions
    initial_report = await generate_initial_report(
        llm,
        state['query'],
        state['findings'],
        extracted_themes,
        report_title,
        state['selected_sources'],
        state.get('formatted_citations', ''),
        current_date,
        state['detail_level'],
        include_objective
    )
    
    # Store the themes for later expansion steps
    state["identified_themes"] = extracted_themes
    state["initial_report"] = initial_report
    log_chain_of_thought(state, "Generated initial comprehensive report with dynamic structure based on content themes")
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state

async def enhance_report_node(llm, progress_callback, state: AgentState) -> AgentState:
    """Enhance the report."""
    state["status"] = "Enhancing report with additional detail"
    console.print("[bold blue]Enhancing report with additional depth and detail...[/]")

    # Use the enhance_report function from processors
    enhanced_report = await enhance_report(
        llm, 
        state["initial_report"], 
        state["current_date"], 
        state.get("formatted_citations", ""),
        state.get("selected_sources", []),
        state["sources"]
    )
    
    state["enhanced_report"] = enhanced_report
    log_chain_of_thought(state, "Enhanced report with additional depth and detail")
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state

async def expand_key_sections_node(llm, progress_callback, state: AgentState) -> AgentState:
    """Expand key sections of the report."""
    state["status"] = "Expanding key sections for maximum depth"
    console.print("[bold blue]Expanding key sections for maximum depth using multi-step synthesis...[/]")

    # Use the expand_key_sections function from processors
    expanded_report = await expand_key_sections(
        llm,
        state["enhanced_report"],
        state["identified_themes"],
        state["current_date"]
    )
    
    state["final_report"] = expanded_report
    log_chain_of_thought(state, "Expanded key sections using multi-step synthesis")
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state

async def report_node(llm, progress_callback, state: AgentState) -> AgentState:
    """Finalize the report."""
    state["status"] = "Finalizing report"
    console.print("[bold blue]Research complete. Finalizing report...[/]")

    # Check if we have any report
    has_report = False
    if "final_report" in state and state["final_report"]:
        final_report = state["final_report"]
        has_report = True
    elif "enhanced_report" in state and state["enhanced_report"]:
        final_report = state["enhanced_report"]
        has_report = True
    elif "initial_report" in state and state["initial_report"]:
        final_report = state["initial_report"]
        has_report = True
    
    # If we have a report but it's broken or too short, regenerate it
    if has_report and (len(final_report.strip()) < 1000):
        console.print("[bold yellow]Existing report appears broken or incomplete. Regenerating...[/]")
        has_report = False
        
    # If we don't have a report, regenerate initial, enhanced, and expanded reports
    if not has_report:
        console.print("[bold yellow]No valid report found. Regenerating report from scratch...[/]")
        
        # Generate a professional title for the report
        report_title = await generate_title(llm, state['query'])
        console.print(f"[bold green]Generated title: {report_title}[/]")
        
        # Extract themes from findings
        extracted_themes = await extract_themes(llm, state['findings'])
        
        # Generate an initial report
        initial_report = await generate_initial_report(
            llm,
            state['query'],
            state['findings'],
            extracted_themes,
            report_title,
            state['selected_sources'],
            state.get('formatted_citations', ''),
            state['current_date'],
            state['detail_level'],
            False # Don't include objective in fallback
        )
        
        # Store the initial report
        state["initial_report"] = initial_report
        
        # Enhance the report
        enhanced_report = await enhance_report(
            llm,
            initial_report,
            state['current_date'],
            state.get('formatted_citations', ''),
            state.get('selected_sources', []),
            state['sources']
        )
        
        # Store the enhanced report
        state["enhanced_report"] = enhanced_report
        
        # Expand key sections
        final_report = await expand_key_sections(
            llm,
            enhanced_report,
            extracted_themes,
            state['current_date']
        )
        
        # Get the sources that were actually analyzed and used in the research
        used_source_urls = []
        for analysis in state["content_analysis"]:
            if "sources" in analysis and isinstance(analysis["sources"], list):
                for url in analysis["sources"]:
                    if url not in used_source_urls:
                        used_source_urls.append(url)
        
        # If we don't have enough used sources, also grab from selected_sources
        if len(used_source_urls) < 5 and "selected_sources" in state:
            for url in state["selected_sources"]:
                if url not in used_source_urls:
                    used_source_urls.append(url)
                    if len(used_source_urls) >= 15:
                        break
        
        # Extract information from sources for the report
        sources_info = []
        for url in used_source_urls[:20]:  # Limit to 20 sources
            source_meta = next((s for s in state["sources"] if s.get("url") == url), {})
            sources_info.append({
                "url": url,
                "title": source_meta.get("title", ""),
                "snippet": source_meta.get("snippet", "")
            })
        
        # Use more specialized processing to handle long report generation in chunks
        console.print("[bold blue]Generating research report in multiple chunks for better reliability...[/]")
        
        # Generate a good title first
        report_title = await generate_title(llm, state['query'])
        console.print(f"[bold green]Report Title: {report_title}[/]")
        
        # Create a structured outline first to guide the detailed content generation
        outline_prompt = f"""Create a detailed outline for a comprehensive research report on {report_title}.
        
        The outline should have:
        1. A clear introduction section
        2. 5-7 main sections (use ## heading level)
        3. 2-3 subsections for each main section (use ### heading level)
        4. A conclusion section
        5. References section
        
        DO NOT write any content - just create the structure with headings.
        Start with '# {report_title}'"""
        
        outline_llm = llm.with_config({"temperature": 0.2})
        outline = outline_llm.invoke(outline_prompt).content
        console.print("[bold green]Generated outline structure[/]")
        
        # Extract sections from the outline to fill them in separately
        sections = re.findall(r'(#+\s+.*?)(?=\n#+\s+|$)', outline, re.DOTALL)
        
        # Generate content for each section in chunks
        full_report = []
        for i, section in enumerate(sections):
            section_title = section.strip().split('\n')[0]
            console.print(f"[bold blue]Generating content for section {i+1}/{len(sections)}: {section_title}[/]")
            
            section_prompt = f"""Write a detailed, comprehensive section for this part of a research report on {report_title}:
            
            {section}
            
            Requirements:
            - Write 800-1200 words of in-depth content for this section
            - Include detailed analysis, examples, and supporting evidence
            - Maintain academic tone and thorough coverage
            - If this is a main section (##), create 2-3 subsections within it (###)
            - If this is the references section, create 15-20 properly formatted references in [n] format
            
            ONLY write content for this specific section - do not rewrite the entire report.
            START with the section heading exactly as shown above.
            """
            
            # Generate content for this section
            section_llm = llm.with_config({"temperature": 0.5})
            try:
                section_content = section_llm.invoke(section_prompt).content
                
                # Clean up section content
                section_content = re.sub(r'^#+\s+.*?\n', f'{section_title}\n', section_content, count=1)
                full_report.append(section_content)
            except Exception as e:
                console.print(f"[bold red]Error generating section {section_title}: {str(e)}[/]")
                # Create a placeholder section if generation fails
                full_report.append(f"{section_title}\n\nThis section content could not be generated.\n\n")
        
        # Combine all sections into a complete report
        final_report = "\n\n".join(full_report)

    # Apply comprehensive cleanup of artifacts and unwanted sections
    final_report = re.sub(r'Completed:.*?\n', '', final_report)
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
    
    # Remove "Based on our discussion" title if it exists
    final_report = re.sub(r'^(?:#\s*)?Based on our discussion,.*?\n', '', final_report, flags=re.MULTILINE)
    
    # Also try to catch Objective sections and other framework components
    final_report = re.sub(r'^Objective:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Key Aspects to Focus On:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Constraints and Preferences:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Areas to Explore in Depth:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Preferred Sources, Perspectives, or Approaches:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Scope, Boundaries, and Context:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    
    # Also remove any remaining individual problem framework lines
    final_report = re.sub(r'^Research Framework:.*?\n', '', final_report, flags=re.MULTILINE)
    final_report = re.sub(r'^Key Findings:.*?\n', '', final_report, flags=re.MULTILINE)
    final_report = re.sub(r'^Key aspects to focus on:.*?\n', '', final_report, flags=re.MULTILINE)
    
    # Ensure the report has the correctly formatted title at the beginning
    report_title = await generate_title(llm, state['query'])
    
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
                console.print("[yellow]Fixing improperly formatted references...[/]")
                
                # Replace problematic references section with properly formatted one
                if state.get("formatted_citations"):
                    new_references = f"# References\n\n{state['formatted_citations']}\n"
                    final_report = final_report.replace(references_section, new_references)
                else:
                    # Generate basic references if formatted_citations isn't available
                    basic_references = []
                    for i, url in enumerate(state.get("selected_sources", []), 1):
                        source_meta = next((s for s in state["sources"] if s.get("url") == url), {})
                        title = source_meta.get("title", "Untitled")
                        source = source_meta.get("source", url.split("//")[1].split("/")[0] if "//" in url else "Unknown Source")
                        date = source_meta.get("date", "n.d.")
                        
                        citation = f"[{i}] *{source}*, \"{title}\", {date}, {url}"
                        basic_references.append(citation)
                    
                    new_references = f"# References\n\n" + "\n".join(basic_references) + "\n"
                    final_report = final_report.replace(references_section, new_references)

    elapsed_time = time.time() - state["start_time"]
    minutes, seconds = divmod(int(elapsed_time), 60)

    state["messages"].append(AIMessage(content="Research complete. Generating final report..."))
    state["findings"] = final_report
    state["status"] = "Complete"

    log_chain_of_thought(state, f"Generated final report after {minutes}m {seconds}s")
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state