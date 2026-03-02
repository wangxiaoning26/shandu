"""
Research agent implementation using LangGraph.
"""  # 文件模块说明：基于 LangGraph 的研究 Agent 实现

import time  # 标准库：用于记录和计算耗时
import asyncio  # 标准库：用于运行异步协程
from datetime import datetime  # 虽然当前文件没直接用到 datetime，但保持统一风格
from typing import List, Dict, Optional, Any, Callable  # 类型标注相关工具

from langchain_openai import ChatOpenAI  # LangChain 的 OpenAI/兼容 LLM 封装
from langchain_core.messages import HumanMessage  # LangChain 核心：人类消息类型
from rich.console import Console  # rich 控制台输出，用于带颜色/布局的终端 UI
from rich.panel import Panel  # rich 的 Panel 组件，用于美观的块状输出

from ..search.search import UnifiedSearcher, SearchResult  # 项目自定义的统一搜索器及其结果类型
from ..scraper import WebScraper, ScrapedContent  # 项目自定义的网页爬虫及其内容结构
from ..research.researcher import ResearchResult  # 统一的研究结果数据结构（用于对外返回）
from ..config import config, get_current_date  # 统一配置读取工具和获取当前日期的函数
from .processors import AgentState  # LangGraph 图中用到的“全局状态”类型（TypedDict）

from .utils.agent_utils import (  # 工具函数：用户输入、查询澄清、进度展示等
    get_user_input,              # 从命令行读取用户输入（当前文件里没直接用到）
    clarify_query,               # 澄清原始查询（当前文件没直接用到，CLI 里会用）
    display_research_progress    # 用 rich 显示研究进度树形结构（CLI 里主要使用）
)

from .nodes import (  # 引入所有“节点函数”，每个节点代表工作流中的一步
    initialize_node,              # 初始化研究（生成研究计划等）
    reflect_node,                 # 反思当前研究结果
    generate_queries_node,        # 生成子查询
    search_node,                  # 执行搜索 + 爬取 + 内容分析
    smart_source_selection,       # 智能选择最相关、可信的来源
    format_citations_node,        # 格式化引用信息（生成文末参考文献等）
    generate_initial_report_node, # 生成报告初稿
    enhance_report_node,          # 增强报告细节
    expand_key_sections_node,     # 扩展关键章节内容
    report_node                   # 生成最终报告文本（收尾）
)

from .graph import build_graph, create_node_wrapper
# build_graph：负责把节点按顺序/条件连成 LangGraph 图
# create_node_wrapper：把普通的异步/同步节点函数包装成 LangGraph 可调用的节点形式

console = Console()  # 初始化一个全局 rich Console，用于本文件的终端输出


class ResearchGraph:
    """Research workflow graph implementation."""
    # 这是一整个“研究工作流”的 OOP 封装：
    # - 负责创建 LLM、Searcher、Scraper
    # - 用 build_graph 搭建 LangGraph 图
    # - 提供 research / research_sync 作为对外接口

    def __init__(
        self, 
        llm: Optional[ChatOpenAI] = None,       # 可选：外部注入自定义 LLM（方便测试或替换模型）
        searcher: Optional[UnifiedSearcher] = None,  # 可选：外部注入搜索器
        scraper: Optional[WebScraper] = None,   # 可选：外部注入爬虫
        temperature: float = 0.5,               # LLM 温度（控制随机性）
        date: Optional[str] = None              # 可选：指定“当前日期”，不传则用 get_current_date()
    ):
        # 从全局 config 中读取模型配置（API 地址 / Key / 模型名）
        api_base = config.get("api", "base_url")  # OpenAI/兼容接口的 base URL
        api_key = config.get("api", "api_key")    # API Key
        model = config.get("api", "model")        # 模型名称（如 gpt-4、hunyuan 等）

        # 初始化 LLM，如果外部没有传入，就用 ChatOpenAI + 配置中参数创建
        self.llm = llm or ChatOpenAI(
            base_url=api_base,        # 使用配置中的 base_url
            api_key=api_key,          # 使用配置中的 api_key
            model=model,              # 使用配置中的模型名
            temperature=temperature,  # 使用传入的温度
            max_tokens=16384          # 单次回答最大 token 数（这里调得很大，支持长报告）
        )

        # 初始化搜索器，如果外部没传就用默认 UnifiedSearcher
        self.searcher = searcher or UnifiedSearcher()

        # 初始化爬虫，如果外部没传就用默认 WebScraper
        self.scraper = scraper or WebScraper()

        # 当前日期：优先用传入的 date，否则调用 get_current_date()
        self.date = date or get_current_date()

        # 进度回调函数，CLI 会传入一个回调用于实时更新 UI，这里先设为 None
        self.progress_callback: Optional[Callable[[AgentState], None]] = None

        # 是否在报告中包含“Objective / 研究目标”章节（由 CLI 参数控制）
        self.include_objective: bool = False

        # 报告的细节级别（目前传给状态，用于节点内决定输出多少细节）
        self.detail_level: str = "high"

        # 构建 LangGraph 图：把各个节点串联成一个完整工作流
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the research graph."""
        # 构建研究工作流图（StateGraph），并编译成可执行图

        # Create wrapped node functions that properly handle async coroutines
        # 说明：节点函数本身需要 LLM、Searcher 等依赖，而 LangGraph 期望的是形如
        #   node(state: AgentState) -> AgentState 的函数
        # 所以这里用 lambda 捕获依赖，然后交给 create_node_wrapper 做必要的异步包装。

        # 初始化节点：负责创建研究计划、写入初始 findings 等
        init_node = create_node_wrapper(
            lambda state: initialize_node(
                self.llm,                # 传入当前 LLM
                self.date,               # 传入当前日期
                self.progress_callback,  # 传入进度回调（用于 UI）
                state                    # 当前 AgentState
            )
        )

        # 反思节点：基于当前 findings 进行高层反思，指导下一轮子问题生成
        reflect = create_node_wrapper(
            lambda state: reflect_node(
                self.llm,                # 需要用 LLM 做反思
                self.progress_callback,  # 更新进度 UI
                state
            )
        )

        # 生成子查询节点：根据反思和当前上下文，生成新的 subqueries
        gen_queries = create_node_wrapper(
            lambda state: generate_queries_node(
                self.llm,                # 用 LLM 把主题拆分为具体子问题
                self.progress_callback,
                state
            )
        )

        # 搜索节点：执行多引擎搜索、爬虫抓取、内容分析
        search = create_node_wrapper(
            lambda state: search_node(
                self.llm,                # 可能需要 LLM 做内容分析
                self.searcher,           # 实际执行搜索的 UnifiedSearcher
                self.scraper,            # 用于抓取网页内容的 WebScraper
                self.progress_callback,
                state
            )
        )

        # 智能选源节点：在所有 sources 中筛选最相关、最可信的那些
        source_selection = create_node_wrapper(
            lambda state: smart_source_selection(
                self.llm,                # 用 LLM 辅助做筛选与排序
                self.progress_callback,
                state
            )
        )

        # 格式化引用节点：将选中的 sources 生成标准引用格式（论文式参考文献等）
        citations = create_node_wrapper(
            lambda state: format_citations_node(
                self.llm,
                self.progress_callback,
                state
            )
        )

        # 生成初稿节点：基于 findings + 选中的 sources + 引用，生成报告初稿
        initial_report = create_node_wrapper(
            lambda state: generate_initial_report_node(
                self.llm,
                self.include_objective,   # 是否包含“Objective”章节
                self.progress_callback,
                state
            )
        )

        # 增强报告节点：在初稿基础上增加细节、补充论证等
        enhance = create_node_wrapper(
            lambda state: enhance_report_node(
                self.llm,
                self.progress_callback,
                state
            )
        )

        # 扩展关键章节节点：进一步展开最关键的部分（如执行摘要、核心结论）
        expand_sections = create_node_wrapper(
            lambda state: expand_key_sections_node(
                self.llm,
                self.progress_callback,
                state
            )
        )

        # 最终报告节点：整理最终报告文本，写入 state["final_report"] / state["findings"]
        final_report = create_node_wrapper(
            lambda state: report_node(
                self.llm,
                self.progress_callback,
                state
            )
        )

        # Build graph with these node functions
        # 把上面的各个节点函数按照既定顺序连接成一个完整 LangGraph 图
        return build_graph(
            init_node,         # 初始化
            reflect,           # 反思
            gen_queries,       # 生成子查询
            search,            # 搜索 + 分析
            source_selection,  # 选源
            citations,         # 格式化引用
            initial_report,    # 初稿
            enhance,           # 增强
            expand_sections,   # 扩展章节
            final_report       # 输出最终报告
        )
        # build_graph 内部会用 StateGraph.add_node / add_edge / add_conditional_edges 等构建有向图，
        # 并最终 compile() 成为 self.graph（CompiledStateGraph）

    async def research(
        self, 
        query: str,                                     # 研究主题 / 顶层问题
        depth: int = 2,                                 # 研究“深度”：多少轮“反思+生成+搜索”
        breadth: int = 4,                               # 每轮最多多少子查询
        progress_callback: Optional[Callable[[AgentState], None]] = None,
        # 可选：自定义 UI 进度回调，CLI 会传一个函数进来

        include_objective: bool = False,                # 是否在报告中包含“Objective”章节
        detail_level: str = "high"                      # 报告细节程度，用于控制节点内部的输出粒度
    ) -> ResearchResult:
        """Execute research process on a query."""
        # 这是异步版的主入口：对外暴露的“研究整个主题”的 API

        # 先把外部传入的配置保存到实例变量中，供各节点在运行中访问
        self.progress_callback = progress_callback      # 用于实时更新进度 UI
        self.include_objective = include_objective      # 决定初稿/最终报告是否包含 Objective
        self.detail_level = detail_level                # 决定输出细节程度

        # 初始化图的起始状态 AgentState（注意：在当前实现中，AgentState 是 TypedDict）
        state = AgentState(
            messages=[HumanMessage(content=f"Starting research on: {query}")],
            # messages：初始消息列表，放入一条“开始研究”的 HumanMessage，部分节点会继续往里追加对话

            query=query,             # 研究主题
            depth=depth,             # 总深度
            breadth=breadth,         # 每层 breadth（用于节点内部取最近 N 个子查询等）
            current_depth=0,         # 当前已经完成的深度，从 0 开始

            findings="",             # 当前研究发现/报告内容的累计文本
            sources=[],              # 所有抓到的来源列表（搜索结果 + 爬取内容摘要）
            selected_sources=[],     # 经过 smart_source_selection 筛选后的来源列表
            formatted_citations="",  # 已格式化的引用（如参考文献列表，Markdown/文本形式）

            subqueries=[],           # 所有生成过的子查询
            content_analysis=[],     # 对网页内容做过的综合分析记录

            start_time=time.time(),  # 记录开始时间，用于统计总耗时
            chain_of_thought=[],     # 用于记录“思考过程”，在 CLI 中可选择性展示

            status="Starting",       # 当前阶段的状态描述字符串（UI 会显示）
            current_date=get_current_date(),  # 当前日期，用于 prompt 中标记“今天是几号”

            detail_level=detail_level,        # 报告细节级别传入状态，节点可根据此值调整粒度

            identified_themes="",    # 后续节点可用于记录“识别出的主题/维度”
            initial_report="",       # 初稿文本（generate_initial_report_node 写入）
            enhanced_report="",      # 增强版报告文本（enhance_report_node 写入）
            final_report=""          # 最终报告文本（report_node 写入）
        )

        # 调用编译好的 LangGraph 图的异步入口：从 entry 节点开始执行，直到 finish 节点结束
        final_state = await self.graph.ainvoke(state)
        # ainvoke：异步执行整个工作流，期间各节点会不断读写 state

        # 计算总耗时
        elapsed_time = time.time() - final_state["start_time"]  # 实际耗时（秒）
        minutes, seconds = divmod(int(elapsed_time), 60)        # 换算成 分钟:秒，仅用于展示

        # 把最终状态整理成 ResearchResult（与 LangChain Agent 方案对齐）
        return ResearchResult(
            query=query,                             # 研究主题
            summary=final_state["findings"],         # 汇总文本（一般包含研究计划 + 各轮 findings + 报告）
            sources=final_state["sources"],          # 所有来源（含结构化字段、可能带详细分析）
            subqueries=final_state["subqueries"],    # 全部子查询列表
            depth=depth,                             # 总深度（即调用时的 depth）

            content_analysis=final_state["content_analysis"],  # 所有内容分析记录
            chain_of_thought=final_state["chain_of_thought"],  # 思考过程（可选展示）

            research_stats={                         # 统计信息：方便 UI 显示，也方便后续分析/日志
                "elapsed_time": elapsed_time,        # 原始耗时（秒）
                "elapsed_time_formatted": f"{minutes}m {seconds}s",  # 人类可读的 mm ss 文本
                "sources_count": len(final_state["sources"]),        # 来源数量
                "subqueries_count": len(final_state["subqueries"]),  # 子查询数量
                "depth": depth,                      # 深度
                "breadth": breadth,                  # 广度
                "detail_level": detail_level         # 报告细节级别
            }
        )

    def research_sync(
        self, 
        query: str,                                 # 研究主题
        depth: int = 2,                             # 深度
        breadth: int = 4,                           # 广度
        progress_callback: Optional[Callable[[AgentState], None]] = None,
        # 可选：进度回调

        include_objective: bool = False,            # 是否包含 Objective
        detail_level: str = "high"                  # 报告细节级别
    ) -> ResearchResult:
        """Synchronous wrapper for research."""
        # 同步包装：为了让 CLI 或其他同步调用方更方便使用，
        # 内部直接用 asyncio.run 调用异步 research()。

        return asyncio.run(
            self.research(               # 调用上面的 async research 方法
                query,                   # 研究主题
                depth,                   # 深度
                breadth,                 # 广度
                progress_callback,       # 进度回调
                include_objective,       # 是否包含 Objective
                detail_level             # 报告细节级别
            )
        )


# These functions are now imported from utils.agent_utils
# 说明：早期版本中，clarify_query / display_research_progress 等函数可能在本文件，
# 现在已迁移到 utils.agent_utils 中，这里只保留这句注释作为迁移记录。
