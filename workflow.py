from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from Nodes.node import *
from States.state import AgentState
import streamlit as st
import sqlite3


# Graph 정의
def create_workflow():
    workflow = StateGraph(AgentState)

    # 노드 정의
    workflow.add_node("Router", select_agent_node)
    workflow.add_node("Search_Agent", product_search_node)
    workflow.add_node("Analysis_Agent", product_analysis_node)
    workflow.add_node("Negotiation_Agent", negotiation_node)

    # router에서 선택된 next_agent 값에 따라 실행할 에이전트 결정
    workflow.add_conditional_edges(
        "Router",
        lambda state: state["next_agent"],
        {
            "Search_Agent": "Search_Agent",
            "Analysis_Agent": "Analysis_Agent",
            "Negotiation_Agent": "Negotiation_Agent",
        },
    )
    # 시작점
    workflow.set_entry_point("Router")

    # 그래프 컴파일
    conn = sqlite3.connect("0701_db", check_same_thread=False)
    memory = SqliteSaver(conn)
    app = workflow.compile(checkpointer=memory)
    return app


def stream_graph(
    app,
    query: str,
    streamlit_container,
    thread_id: str,
):
    config = RunnableConfig(recursion_limit=10, configurable={"thread_id": thread_id})
    # AgentState 객체를 활용하여 질문을 입력합니다.
    inputs = AgentState(question = query, next_agent = "Search_Agent", generation = "")

    # app.stream을 통해 입력된 메시지에 대한 출력을 스트리밍합니다.
    actions = {
        "Router": "🤔 질문의 의도를 분석하는 중입니다.",
        "Search_Agent": "🔍 제품을 탐색/추천하는 중입니다.",
        "Analysis_Agent": "📊 제품을 비교/분석하는 중입니다.",
        "Negotiation_Agent": "🤝 판매자와 협상하는 중입니다.",
    }

    try:
        # streamlit_container
        with streamlit_container.status(
            "😊 열심히 생각중 입니다...", expanded=True
        ) as status:
            # st.write("🧑‍💻 질문의 의도를 분석하는 중입니다.")
            for output in app.stream(inputs, config=config):
                # 출력된 결과에서 키와 값을 순회합니다.
                for key, value in output.items():
                    # 노드의 이름과 해당 노드에서 나온 출력을 출력합니다.
                    if key in actions:
                        st.write(actions[key])
                # 출력 값을 예쁘게 출력합니다.
            status.update(label="답변 완료", state="complete", expanded=False)
    except Exception as e:
        print(f"Error during streaming: {e}")
    return app.get_state(config={"configurable": {"thread_id": thread_id}}).values
