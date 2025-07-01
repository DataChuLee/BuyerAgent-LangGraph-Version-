from langchain_core.messages import BaseMessage
from typing import Sequence, Annotated, Literal, List, Dict, Optional
from typing_extensions import TypedDict
import functools, operator

class AgentState(TypedDict):
    """
    그래프의 상태를 나타내는 데이터 모델

    Attributes:
        input: 질문
        next_agent: 다음 에이전트의 이름 (Search_Agent, Analysis_Agent, Negotiation_Agent)
        generation: 각각의 Agent가 생성한 답변
    """

    question: Annotated[str, "User question"]  # 사용자의 질문
    generation: Annotated[str, "Agent generated answer"]  # AI의 답변
    next_agent: Literal["Search_Agent", "Analysis_Agent", "Negotiation_Agent"]
