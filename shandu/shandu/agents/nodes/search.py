"""
Search node for research graph.
Handles searching and processing of content.
"""
import time
import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from langchain_core.messages import HumanMessage
from ..processors.content_processor import AgentState, is_relevant_url, process_scraped_item, analyze_content
from ..utils.agent_utils import log_chain_of_thought, _call_progress_callback

console = Console()

async def search_node(llm, searcher, scraper, progress_callback, state: AgentState) -> AgentState:
    """Search for information and analyze results for the current research queries."""
    state["status"] = "Searching and analyzing content"
    recent_queries = state["subqueries"][-state["breadth"]:]
    processed_queries = set()
    
    # Create a simple LRU cache for URL relevance checks to reduce redundant LLM calls
    url_relevance_cache = {}
    max_cache_size = 100
    
    # Create a wrapper around is_relevant_url to add caching
    async def cached_relevance_check(url, title, snippet, query):
        # Check cache before making LLM call
        cache_key = f"{url}:{query}"
        if cache_key in url_relevance_cache:
            return url_relevance_cache[cache_key]
        
        # Call the imported is_relevant_url function
        result = await is_relevant_url(llm, url, title, snippet, query)
        
        # Update cache with result
        if len(url_relevance_cache) >= max_cache_size:
            # Remove a random item if cache is full
            url_relevance_cache.pop(next(iter(url_relevance_cache)))
        url_relevance_cache[cache_key] = result
        
        return result
    
    # Process multiple queries in parallel
    async def process_query(subquery, query_task, progress):
        if subquery in processed_queries:
            return
        
        processed_queries.add(subquery)
        
        try:
            log_chain_of_thought(state, f"Searching for: {subquery}")
            console.print(f"[dim]Executing search for: {subquery}[/dim]")
            search_results = await searcher.search(subquery)
            progress.update(query_task, advance=0.3, description=f"[yellow]Found {len(search_results)} results for: {subquery}")
            
            urls = []
            seen = set()
            
            # Batch URL relevance checks with asyncio.gather
            relevance_tasks = []
            for i, result in enumerate(search_results):
                if len(urls) >= 5:  # Maximum number of URLs to analyze
                    break
                if (result.url and isinstance(result.url, str) and result.url not in seen and 
                    result.url.startswith('http')):
                    relevance_tasks.append((i, result, cached_relevance_check(result.url, result.title, result.snippet, subquery)))
            
            # Wait for all relevance checks to complete
            for i, result, relevance_task in relevance_tasks:
                is_relevant = await relevance_task
                if is_relevant and len(urls) < 5:
                    urls.append(result.url)
                    seen.add(result.url)
                    log_chain_of_thought(state, f"Selected relevant URL: {result.url}")
                    console.print(f"[green]Selected for analysis:[/green] {result.title}")
                    console.print(f"[blue]URL:[/blue] {result.url}")
                    if result.snippet:
                        console.print(f"[dim]{result.snippet[:150]}{'...' if len(result.snippet) > 150 else ''}[/dim]")
                    console.print("")
            
            if urls:
                progress.update(query_task, advance=0.2, description=f"[yellow]Scraping {len(urls)} pages for: {subquery}")
                scraped = await scraper.scrape_urls(urls, dynamic=True)
                successful_scraped = [s for s in scraped if s.is_successful()]
                
                if successful_scraped:
                    progress.update(query_task, advance=0.2, description=f"[yellow]Analyzing content for: {subquery}")
                    content_text = ""
                    
                    # Process all scraped items in parallel using our processor module
                    processing_tasks = []
                    for item in successful_scraped:
                        main_content = await scraper.extract_main_content(item)
                        processing_tasks.append(process_scraped_item(llm, item, subquery, main_content))
                    
                    # Process all items in parallel
                    processed_items = await asyncio.gather(*processing_tasks)
                    
                    # Build content text from processed items
                    relevant_items = []
                    for processed in processed_items:
                        if processed["rating"] == "LOW":
                            console.print(f"[yellow]Skipping low-reliability source: {processed['item'].url}[/yellow]")
                            continue
                        
                        relevant_items.append(processed)
                        content_text += f"\nSource: {processed['item'].url}\nTitle: {processed['item'].title}\nReliability: {processed['rating']}\nRelevant Content:\n{processed['content']}\n\n"
                        console.print(f"\n[bold cyan]Analyzing page:[/bold cyan] {processed['item'].title}")
                        console.print(f"[blue]URL:[/blue] {processed['item'].url}")
                        console.print(f"[dim]Extracted Content: {processed['content'][:150]}{'...' if len(processed['content']) > 150 else ''}[/dim]")
                    
                    if relevant_items:
                        try:
                            # Use our analyze_content function from the processors module
                            analysis_content = await analyze_content(llm, subquery, content_text)
                            
                            state["content_analysis"].append({
                                "subquery": subquery,
                                "analysis": analysis_content,
                                "sources": [item["item"].url for item in relevant_items]
                            })
                            
                            # Store findings with thematic headers
                            state["findings"] += f"\n\n## Research on '{subquery}':\n{analysis_content}\n"
                            log_chain_of_thought(state, f"Analyzed content for query: {subquery}")
                        except Exception as e:
                            console.print(f"[dim red]Error in content analysis for {subquery}: {str(e)}[/dim red]")
                            # Log the error but continue with the process
                            log_chain_of_thought(state, f"Error analyzing content for '{subquery}': {str(e)}")
            
            for r in search_results:
                if hasattr(r, 'to_dict'):
                    state["sources"].append(r.to_dict())
                elif isinstance(r, dict):
                    state["sources"].append(r)
            
            progress.update(query_task, completed=True, description=f"[green]Completed: {subquery}")
            return True
            
        except Exception as e:
            progress.update(query_task, completed=True, description=f"[red]Error: {subquery} - {str(e)}")
            state["messages"].append(HumanMessage(content=f"Failed to process subquery: {subquery}"))
            log_chain_of_thought(state, f"Error processing query '{subquery}': {str(e)}")
            console.print(f"[dim red]Error processing {subquery}: {str(e)}[/dim red]")
            return False
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        # Create all tasks first
        tasks = {}
        for i, subquery in enumerate(recent_queries):
            if subquery not in processed_queries:
                task_id = progress.add_task(f"[yellow]Searching: {subquery}", total=1)
                tasks[subquery] = task_id
        
        # Process queries in parallel batches to avoid overwhelming the system
        batch_size = min(3, len(tasks))  # Process up to 3 queries at once
        
        for i in range(0, len(tasks), batch_size):
            batch_queries = list(tasks.keys())[i:i+batch_size]
            batch_tasks = [process_query(query, tasks[query], progress) for query in batch_queries]
            await asyncio.gather(*batch_tasks)
    
    state["current_depth"] += 1
    elapsed_time = time.time() - state["start_time"]
    minutes, seconds = divmod(int(elapsed_time), 60)
    state["status"] = f"Completed depth {state['current_depth']}/{state['depth']} ({minutes}m {seconds}s elapsed)"
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state