"""
Citation formatting node for research graph.
"""
from rich.console import Console
from ..processors.content_processor import AgentState
from ..processors.report_generator import format_citations
from ..utils.agent_utils import log_chain_of_thought, _call_progress_callback

console = Console()

async def format_citations_node(llm, progress_callback, state: AgentState) -> AgentState:
    """Format citations for selected sources to ensure consistent referencing."""
    state["status"] = "Formatting citations"
    console.print("[bold blue]Formatting source citations...[/]")
    
    selected_urls = state["selected_sources"]
    if not selected_urls:
        log_chain_of_thought(state, "No sources selected for citations")
        return state
    
    # Use the format_citations function from processors
    formatted_citations = await format_citations(llm, selected_urls, state["sources"])
    
    state["formatted_citations"] = formatted_citations
    log_chain_of_thought(state, f"Formatted citations for {len(selected_urls)} sources")
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state