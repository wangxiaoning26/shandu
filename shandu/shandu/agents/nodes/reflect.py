"""
Reflection node for research graph.
"""
from rich.console import Console
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from ..processors.content_processor import AgentState
from ..utils.agent_utils import log_chain_of_thought, _call_progress_callback
from ...prompts import SYSTEM_PROMPTS, USER_PROMPTS

console = Console()

async def reflect_node(llm, progress_callback, state: AgentState) -> AgentState:
    """Reflect on current findings to identify gaps and opportunities."""
    state["status"] = "Reflecting on findings"
    console.print("[bold yellow]Reflecting on current findings...[/]")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPTS["reflection"].format(current_date=state["current_date"])),
        ("user", USER_PROMPTS["reflection"].format(findings=state["findings"]))
    ])
    
    chain = prompt | llm
    reflection = chain.invoke({"findings": state["findings"]})
    
    state["messages"].append(HumanMessage(content="Analyzing current findings..."))
    state["messages"].append(AIMessage(content=reflection.content))
    state["findings"] += f"\n\n## Reflection on Current Findings\n\n{reflection.content}\n\n"
    
    log_chain_of_thought(state, "Completed reflection on current findings")
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state