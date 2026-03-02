"""Agent module for Shandu research system."""
from typing import List, Dict, Optional, Union, Any  # 类型标注，用于提高可读性和 IDE 提示
from dataclasses import dataclass                   # dataclass，项目里有用到（比如 ResearchResult）
from datetime import datetime                       # 时间工具，目前本文件没直接用到，但统一风格
import asyncio                                      # 用于把 async research 封装成同步调用
import json                                         # 如果以后要序列化数据会用到
import time                                         # 预留给计时等用途

from langchain_core.prompts import ChatPromptTemplate   # LangChain 核心：多轮对话形式的 Prompt 模板
from langchain_core.output_parsers import StrOutputParser  # 把 LLM 返回的消息解析成纯字符串
from langchain_core.runnables import RunnablePassthrough   # 这里没用到，但可用于构建更复杂的链
from langchain_openai import ChatOpenAI                # OpenAI/兼容接口的 Chat LLM 封装
from langchain.agents import AgentType, initialize_agent  # LangChain Agent 类型和工厂方法
from langchain.chains import LLMChain                  # 传统链式调用（本文件没直接用到）
from langchain.prompts import PromptTemplate           # 传统单轮文本 Prompt 模板
from langchain_community.tools import (                # LangChain 内置/社区工具
    Tool,                                              # 通用 Tool 包装器
    DuckDuckGoSearchResults,                           # DuckDuckGo 返回多条结果的工具
    DuckDuckGoSearchRun                                # DuckDuckGo 简单搜索工具（返回单条/直接答案）
)

from ..search.search import UnifiedSearcher, SearchResult  # 项目自己的统一搜索封装 & 结果类型
from ..research.researcher import ResearchResult           # 统一的研究结果数据结构
from ..scraper import WebScraper, ScrapedContent           # 网页爬取与内容结构
from ..prompts import SYSTEM_PROMPTS, USER_PROMPTS         # 集中管理的系统 / 用户 Prompt 模板


class ResearchAgent:
    """LangChain-based research agent."""
    # 说明：这是基于 LangChain Agent 的“研究智能体”，提供 research / research_sync 两个入口。

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,          # 可注入自定义 LLM（便于测试或换模型）
        searcher: Optional[UnifiedSearcher] = None,# 可注入自定义搜索实现（默认用项目内 UnifiedSearcher）
        scraper: Optional[WebScraper] = None,      # 可注入自定义爬虫（默认用项目内 WebScraper）
        temperature: float = 0,                    # LLM 温度，0 更偏“确定性”
        max_depth: int = 2,                        # 研究“深度”——递进轮数
        breadth: int = 4,                          # 每一轮生成多少个子查询
        max_urls_per_query: int = 3,               # 每个子查询最多抓多少个 URL
        proxy: Optional[str] = None                # 爬虫使用的代理（可选）
    ):
        # Initialize components
        # 初始化底层组件：LLM、搜索器、爬虫等
        self.llm = llm or ChatOpenAI(
            temperature=temperature,               # 使用传入的温度
            model="hunyuan-standard"               # 默认模型名（注释里写 GPT-4，但这里可接入任何兼容模型）
        )
        self.searcher = searcher or UnifiedSearcher()        # 如果外部没传，就用默认的统一搜索器
        self.scraper = scraper or WebScraper(proxy=proxy)    # 如果外部没传，就用默认的爬虫，并配置代理

        # Research parameters
        # 研究参数，用于控制算法行为
        self.max_depth = max_depth                            # 最多迭代几轮深度研究
        self.breadth = breadth                                # 每轮最多多少个子问题
        self.max_urls_per_query = max_urls_per_query          # 每个子问题最多抓多少个网页

        # Initialize prompts
        # 初始化会用到的 Prompt 模板，来源于项目统一的 prompts 配置
        self.system_prompt = ChatPromptTemplate.from_template(
            SYSTEM_PROMPTS["research_agent"]                  # 针对“研究 Agent”的系统级指令
        )
        self.reflection_prompt = ChatPromptTemplate.from_template(
            USER_PROMPTS["reflection"]                        # 用于“反思当前研究结果”的用户 Prompt
        )
        self.query_gen_prompt = ChatPromptTemplate.from_template(
            USER_PROMPTS["query_generation"]                  # 用于“生成新的子查询”的用户 Prompt
        )

        # Initialize tools
        # 初始化该 Agent 可用的工具列表（LangChain Tool）
        self.tools = self._setup_tools()

        # Initialize agent
        # 用 LangChain 提供的 initialize_agent 创建一个“自动选择工具”的智能体
        self.agent = initialize_agent(
            tools=self.tools,                                 # 上面定义的工具列表
            llm=self.llm,                                     # 底层大模型
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            # Agent 类型：支持结构化聊天 + 基于工具描述的 Zero-shot 推理
            verbose=True                                      # 打印详细的中间推理/调用过程（方便调试）
        )

    def _setup_tools(self) -> List[Tool]:
        """Setup agent tools."""
        # 说明：把项目里的搜索 / 反思 / 子查询生成能力包装成 LangChain Tool，
        # 供 self.agent 在运行时按需调用。
        return [
            Tool(
                name="search",                                # 工具名（供 LLM 在推理中引用）
                func=self.searcher.search_sync,               # 实际 Python 函数：同步搜索接口
                description="Search multiple sources for information about a topic"
                # 描述：告诉 Agent 这个工具能干什么，直接影响 LLM 的工具选择
            ),
            DuckDuckGoSearchResults(
                name="ddg_results",                           # 内置 DuckDuckGo 工具：多结果
                description="Get detailed search results from DuckDuckGo"
            ),
            DuckDuckGoSearchRun(
                name="ddg_search",                            # 内置 DuckDuckGo 工具：快速搜索
                description="Search DuckDuckGo for a quick answer"
            ),
            Tool(
                name="reflect",                               # 自定义的“反思”工具
                func=self._reflect_on_findings,               # 绑定到本类的异步方法
                description="Analyze and reflect on current research findings"
            ),
            Tool(
                name="generate_queries",                      # 自定义的“生成子查询”工具
                func=self._generate_subqueries,               # 绑定到本类的异步方法
                description="Generate targeted subqueries for deeper research"
            )
        ]

    async def _reflect_on_findings(self, findings: str) -> str:
        """Analyze research findings."""
        # 说明：对当前的研究结果文本做一轮 LLM 反思，找出不足与下一步方向。
        reflection_chain = self.reflection_prompt | self.llm | StrOutputParser()
        # 上面这一行：构建一个链 = (反思 Prompt) -> (LLM) -> (输出解析为字符串)
        return await reflection_chain.ainvoke({"findings": findings})
        # 异步调用链，传入当前 findings 文本，返回字符串形式的反思结果

    async def _generate_subqueries(
        self,
        query: str,          # 原始顶层研究问题
        findings: str,       # 当前已经得到的研究结果
        questions: str       # 反思阶段提出的疑问/关注点
    ) -> List[str]:
        """Generate subqueries for deeper research."""
        # 说明：基于当前成果和反思，生成一批新的子问题，用于驱动下一轮搜索。
        query_chain = self.query_gen_prompt | self.llm | StrOutputParser()
        # 构建“生成子查询”的链
        result = await query_chain.ainvoke({
            "query": query,          # 顶层问题
            "findings": findings,    # 当前 findings
            "questions": questions,  # 来自反思阶段的未解问题
            "breadth": self.breadth  # 希望生成的子问题数量（由 Prompt 决定如何使用）
        })

        # Parse the result into a list of queries
        # 说明：假定 LLM 一行一个子问题，这里按行拆分并做简单清洗
        queries = [q.strip() for q in result.split("\n") if q.strip()]
        return queries[:self.breadth]  # 最多保留 breadth 个子问题

    async def _extract_urls_from_results(
        self,
        search_results: List[SearchResult],   # 搜索结果列表（项目自定义类型）
        max_urls: int = 3                    # 最多提取多少 URL
    ) -> List[str]:
        """Extract top URLs from search results."""
        # 说明：从搜索结果中提取去重后的前若干个有效 URL
        urls = []                            # 存储最终选出的 URL
        seen = set()                         # 用于 URL 去重

        for result in search_results:
            if len(urls) >= max_urls:        # 数量达到上限就提前结束
                break

            url = result.url                 # 取出搜索结果中的 URL
            if url and url not in seen and url.startswith('http'):
                # 条件：非空、没出现过、是 http 开头的正常链接
                urls.append(url)             # 收集该 URL
                seen.add(url)                # 标记为已见

        return urls                          # 返回最终 URL 列表

    async def _analyze_content(
        self,
        query: str,                          # 当前子查询或主查询
        content: List[ScrapedContent]        # 爬虫抓到的网页内容列表
    ) -> Dict[str, Any]:
        """Analyze scraped content."""
        # 说明：把多个网页内容拼接起来，交给 LLM 做“综合内容分析”。

        # Prepare content for analysis
        # 把各个网页的标题、URL、截断后的正文摘要拼成一段大文本
        content_text = ""
        for item in content:
            content_text += f"\nSource: {item.url}\nTitle: {item.title}\n"
            content_text += f"Content Summary:\n{item.text[:1000]}...\n"
            # 为了防止 prompt 过长，这里只截取前 1000 字符，并加上省略号

        # Use the content analysis prompt from centralized prompts
        # 说明：这里没有直接用 ChatPromptTemplate.from_template，而是用 from_messages 来构建 system+user
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPTS["content_analysis"]),
            # 系统消息：告诉模型要如何分析多源内容
            ("user", USER_PROMPTS["content_analysis"].format(
                query=query,
                content=content_text
            ))
            # 用户消息：把当前 query 和拼好的 content_text 塞到统一的模板里
        ])

        analysis_chain = analysis_prompt | self.llm | StrOutputParser()
        # 构建“内容分析”链（Prompt -> LLM -> 字符串解析）
        analysis = await analysis_chain.ainvoke({})
        # 这里没有额外输入变量，因为 user prompt 已经在 format 里植入了 query 和 content

        return {
            "analysis": analysis,                # LLM 生成的综合分析文本
            "sources": [c.url for c in content]  # 本轮分析涉及到的所有来源 URL
        }

    async def research(
        self,
        query: str,                               # 顶层研究问题
        depth: Optional[int] = None,             # 指定研究深度（不传则用默认 self.max_depth）
        engines: List[str] = ["google", "duckduckgo"]  # 使用哪些搜索引擎
    ) -> ResearchResult:
        """Execute the research process."""
        # 说明：这是异步版的研究主流程，负责 orchestrate（编排）各步骤，
        # 返回统一的 ResearchResult 结构给上层使用。

        depth = depth if depth is not None else self.max_depth
        # 如果调用者传了 depth，就用传入的；否则用默认 max_depth

        # Initialize research context
        # 构造一个“上下文字典”，在整个研究过程中持续累加信息
        context = {
            "query": query,              # 顶层问题
            "depth": depth,              # 研究深度
            "breadth": self.breadth,     # 每轮子问题数量
            "findings": "",              # 当前所有研究发现/说明文本
            "sources": [],               # 所有来源（搜索结果、URL 等）
            "subqueries": [],            # 所有生成过的子问题
            "content_analysis": []       # 对网页内容做过的深度分析记录
        }

        # Initial system prompt to set up the research
        # 第一步：用系统级 Prompt 对 query 做一个“开场白/研究框架”，写入 findings
        system_chain = self.system_prompt | self.llm | StrOutputParser()
        context["findings"] = await system_chain.ainvoke(context)
        # 这里把 context 整体传入，Prompt 内会用到 query/depth 等字段

        # Iterative deepening research process
        # 说明：按 depth 层次循环，每一层都做“反思 -> 生成子问题 -> 搜索+分析 -> 更新上下文”
        for current_depth in range(depth):
            # Reflect on current findings
            # 1）对当前 findings 做“反思”，找出哪里还不够、还有哪些问题
            reflection = await self._reflect_on_findings(context["findings"])

            # Generate new subqueries based on reflection
            # 2）基于反思结果生成新的子查询列表
            new_queries = await self._generate_subqueries(
                query=query,
                findings=context["findings"],
                questions=reflection
            )
            context["subqueries"].extend(new_queries)   # 追加到全局子查询列表中

            # Search and analyze for each new query
            # 3）对每一个新子查询执行“智能 Agent 调用 + 真实搜索 + 内容分析”
            for subquery in new_queries:
                # Use the agent to decide how to approach this subquery
                # 3.1 使用 LangChain Agent 自主决定如何使用工具（search / ddg_search / reflect 等）
                agent_result = await self.agent.arun(
                    f"Research this specific aspect: {subquery}\n\n"
                    f"Current findings: {context['findings']}\n\n"
                    "Think step by step about what tools to use and how to verify the information."
                )
                # 这里传给 Agent 的是自然语言指令，它会根据工具描述自动选择调用顺序

                # Perform the search
                # 3.2 再用我们自己的 UnifiedSearcher 做一次“真实搜索”
                search_results = await self.searcher.search(
                    subquery,
                    engines=engines
                )

                # Extract URLs for scraping
                # 3.3 从搜索结果中提取有限个 URL，用于后续爬取
                urls_to_scrape = await self._extract_urls_from_results(
                    search_results,
                    self.max_urls_per_query
                )

                # Scrape and analyze content
                # 3.4 如果有 URL，就用爬虫抓取网页，并对内容进行分析
                if urls_to_scrape:
                    scraped_content = await self.scraper.scrape_urls(
                        urls_to_scrape,
                        dynamic=True     # 开启动态渲染（适合 JS-heavy 的网站）
                    )

                    if scraped_content:
                        # Analyze the content
                        # 3.5 调用 _analyze_content，对多源网页内容做 LLM 综合分析
                        analysis = await self._analyze_content(subquery, scraped_content)
                        context["content_analysis"].append({
                            "subquery": subquery,
                            "analysis": analysis["analysis"],
                            "sources": analysis["sources"]
                        })

                # Add results to context
                # 3.6 把搜索结果整合进 context["sources"]
                for r in search_results:
                    if isinstance(r, SearchResult):
                        context["sources"].append(r.to_dict())
                    elif isinstance(r, dict):
                        context["sources"].append(r)
                    else:
                        print(f"Warning: Skipping non-serializable search result: {type(r)}")

                # 把 Agent 返回的文字结果附加到 findings 中，作为该子问题的发现记录
                context["findings"] += f"\n\nFindings for '{subquery}':\n{agent_result}"

                # Add content analysis if available
                # 如果刚刚有内容分析结果，则也 append 到 findings 中
                if context["content_analysis"]:
                    latest_analysis = context["content_analysis"][-1]
                    context["findings"] += f"\n\nDetailed Analysis:\n{latest_analysis['analysis']}"

        # Final reflection and summary
        # 说明：所有 depth 轮结束后，再做一次“全局反思”，作为最终 summary
        final_reflection = await self._reflect_on_findings(context["findings"])

        # Prepare detailed sources with content analysis
        # 说明：为每个 source 关联对应的 content_analysis，形成“带详细分析的来源列表”
        detailed_sources = []
        for source in context["sources"]:
            # Source is already a dictionary at this point
            source_dict = source.copy()  # Make a copy to avoid modifying the original

            # Add any content analysis related to this source
            # 遍历所有 content_analysis，凡是该 source 的 URL 在其中出现，就把分析文本挂到 source 上
            for analysis in context["content_analysis"]:
                if source.get("url", "") in analysis["sources"]:
                    source_dict["detailed_analysis"] = analysis["analysis"]

            detailed_sources.append(source_dict)

        # 把所有累积的信息封装成统一的 ResearchResult 返回给上层
        return ResearchResult(
            query=query,                          # 顶层问题
            summary=final_reflection,             # 最终总结（来自最后一次反思）
            sources=detailed_sources,             # 带详细分析的来源列表
            subqueries=context["subqueries"],     # 全部子问题
            depth=depth,                          # 实际使用的深度
            content_analysis=context["content_analysis"]  # 所有内容分析记录
        )

    def research_sync(
        self,
        query: str,                               # 顶层研究问题
        depth: Optional[int] = None,              # 可选：指定深度
        engines: List[str] = ["google", "duckduckgo"]  # 搜索引擎列表
    ) -> ResearchResult:
        """Synchronous research wrapper."""
        # 说明：同步封装，方便 CLI 或其他同步环境调用，
        # 内部通过 asyncio.run 调用上面的 async research。
        return asyncio.run(self.research(query, depth, engines))
