from Agents.agent import (
    router,
    product_search_agent,
    product_analysis_agent,
    negotiation_agent,
)
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessage
from typing import Optional, Literal
from States.state import AgentState


# State 논리 정의
def select_agent_node(state: AgentState) -> AgentState:
    question = state["question"]
    router_chain = router()
    next_agent = router_chain.invoke(
        {"question": question}
    )  # "search" / "analysis" / "negotiation"

    # 반환값이 올바른지 확인하고, 아니면 기본값 지정
    if next_agent not in ["Search_Agent", "Analysis_Agent", "Negotiation_Agent"]:
        print(f"[경고] router_chain이 이상한 값을 반환함: {next_agent}, 기본값으로 Search_Agent 사용")
        next_agent = "Search_Agent"

    return AgentState(question = question, next_agent = next_agent, generation = "")


def product_search_node(state: AgentState) -> AgentState:
    question = state["question"]
    product_search_agent_executor = product_search_agent()
    response = product_search_agent_executor.invoke({"question": question})
    return AgentState(
        question = question, next_agent=state["next_agent"], generation=response["output"]
    )


def product_analysis_node(state: AgentState) -> AgentState:
    question = state["question"]
    product_analysis_agent_executor = product_analysis_agent()
    response = product_analysis_agent_executor.invoke({"question": question})
    return AgentState(
        question = question, next_agent=state["next_agent"], generation=response["output"]
    )


def negotiation_node(state: AgentState) -> AgentState:
    question = state["question"]
    negotiation_agent_executor = negotiation_agent()
    response = negotiation_agent_executor.invoke({"question": question})
    return AgentState(
        question = question, next_agent=state["next_agent"], generation=response["output"]
    )
