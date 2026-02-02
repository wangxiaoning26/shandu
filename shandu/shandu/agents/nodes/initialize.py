"""
Initialize node for research graph.
"""
import time
from rich.console import Console
from rich.panel import Panel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from ..processors.content_processor import AgentState
from ..utils.agent_utils import log_chain_of_thought, _call_progress_callback
from ...config import get_current_date
from ...prompts import SYSTEM_PROMPTS, USER_PROMPTS

console = Console()

async def initialize_node(llm, date, progress_callback, state: AgentState) -> AgentState:
    """Initialize the research process with a research plan."""
    console.print(Panel(f"[bold blue]Starting Research:[/] {state['query']}", title="Research Process", border_style="blue"))
    state["start_time"] = time.time()
    state["status"] = "Initializing research"
    state["current_date"] = date or get_current_date()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPTS["initialize"].format(current_date=state["current_date"])),
        ("user", USER_PROMPTS["initialize"].format(query=state["query"]))
    ])
    
    chain = prompt | llm
    plan = chain.invoke({"query": state["query"]})
    
    # Clean up any markdown formatting that might have been included
    cleaned_plan = plan.content.replace("**", "").replace("# ", "").replace("## ", "")
    
    state["messages"].append(HumanMessage(content=f"Planning research on: {state['query']}"))
    state["messages"].append(AIMessage(content=cleaned_plan))
    state["findings"] = f"# Research Plan\n\n{cleaned_plan}\n\n# Initial Findings\n\n"
    
    log_chain_of_thought(state, f"Created research plan for query: {state['query']}")
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state