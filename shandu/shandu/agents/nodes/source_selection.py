"""Source selection node."""
from rich.console import Console
from langchain_core.prompts import ChatPromptTemplate
from ..processors.content_processor import AgentState
from ..utils.agent_utils import log_chain_of_thought, _call_progress_callback
from ...prompts import SYSTEM_PROMPTS, USER_PROMPTS

console = Console()

async def smart_source_selection(llm, progress_callback, state: AgentState) -> AgentState:
    """Select relevant sources for the report."""
    state["status"] = "Selecting most valuable sources"
    console.print("[bold blue]Selecting most relevant and high-quality sources...[/]")
    
    # Get all sources found during research
    all_source_urls = []
    for analysis in state["content_analysis"]:
        if "sources" in analysis and isinstance(analysis["sources"], list):
            for url in analysis["sources"]:
                if url not in all_source_urls:
                    all_source_urls.append(url)
    
    # If we have too many sources, use smart selection to filter them
    if len(all_source_urls) > 25:
        # Format the sources to be evaluated
        sources_text = ""
        for i, url in enumerate(all_source_urls, 1):
            # Find the source metadata if available
            source_meta = {}
            for source in state["sources"]:
                if source.get("url") == url:
                    source_meta = source
                    break
            
            # Add source information to the text
            sources_text += f"Source {i}:\nURL: {url}\n"
            if source_meta.get("title"):
                sources_text += f"Title: {source_meta.get('title')}\n"
            if source_meta.get("snippet"):
                sources_text += f"Summary: {source_meta.get('snippet')}\n"
            if source_meta.get("date"):
                sources_text += f"Date: {source_meta.get('date')}\n"
            sources_text += "\n"
        
        # Use the smart source selection prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPTS["smart_source_selection"]),
            ("user", USER_PROMPTS["smart_source_selection"].format(
                query=state["query"],
                sources=sources_text
            ))
        ])
        
        selection_chain = prompt | llm
        selection_result = selection_chain.invoke({})
        
        # Extract the selected sources from the result
        selected_urls = []
        for url in all_source_urls:
            if url in selection_result.content:
                selected_urls.append(url)
        
        # Make sure we have at least some sources
        if not selected_urls and all_source_urls:
            # If selection failed, take the first 15-20 sources
            selected_urls = all_source_urls[:min(20, len(all_source_urls))]
            
        state["selected_sources"] = selected_urls
        log_chain_of_thought(state, f"Selected {len(selected_urls)} most relevant sources from {len(all_source_urls)} total sources")
    else:
        # If we don't have too many sources, use all of them
        state["selected_sources"] = all_source_urls
        log_chain_of_thought(state, f"Using all {len(all_source_urls)} sources for final report")
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state