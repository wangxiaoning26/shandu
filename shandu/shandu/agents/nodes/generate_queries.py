"""
Query generation node for research graph.
"""
import re
from rich.console import Console
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from ..processors.content_processor import AgentState
from ..utils.agent_utils import log_chain_of_thought, _call_progress_callback
from ...prompts import SYSTEM_PROMPTS, USER_PROMPTS

console = Console()

async def generate_queries_node(llm, progress_callback, state: AgentState) -> AgentState:
    """Generate targeted search queries based on current findings."""
    state["status"] = "Generating research queries"
    console.print("[bold yellow]Generating targeted search queries...[/]")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPTS["query_generation"].format(current_date=state["current_date"])),
        ("user", USER_PROMPTS["query_generation"].format(
            query=state["query"], 
            findings=state["findings"], 
            breadth=state["breadth"]
        ))
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": state["query"], "findings": state["findings"], "breadth": state["breadth"]})
    
    # Process the generated queries
    # Extract direct search queries without any formatting or headers
    new_queries = [line.strip() for line in result.content.split("\n") if line.strip()]
    # Remove any numbering, bullet points, or other formatting
    new_queries = [re.sub(r'^[\d\s\-\*•\.\)]+\s*', '', line).strip() for line in new_queries]
    # Remove phrases like "Here are...", "I'll search for..." etc.
    new_queries = [re.sub(r'^(here are|i will|i\'ll|let me|these are|i recommend|completed:|search for:).*?:', '', line, flags=re.IGNORECASE).strip() for line in new_queries]
    # Filter out any empty lines or lines that don't look like actual queries
    new_queries = [q for q in new_queries if q and len(q.split()) >= 2 and not q.lower().startswith(("query", "search", "investigate", "explore", "research"))]
    # Limit to the specified breadth
    new_queries = new_queries[:state["breadth"]]
    
    state["messages"].append(HumanMessage(content="Generating new research directions..."))
    state["messages"].append(AIMessage(content="Generated queries:\n" + "\n".join(new_queries)))
    state["subqueries"].extend(new_queries)
    
    console.print("[bold green]Generated search queries:[/]")
    for i, query in enumerate(new_queries, 1):
        console.print(f"  {i}. {query}")
    
    log_chain_of_thought(state, f"Generated {len(new_queries)} search queries for investigation")
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state