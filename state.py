from langchain_core.messages import BaseMessage
from typing import Sequence, Annotated, Literal, List, Dict, Optional
from typing_extensions import TypedDict
import functools, operator


# # State 정의
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], operator.add]  # 메시지
#     next_agent: Optional[
#         Literal["Search_Agent", "Analysis_Agent", "Negotiation_Agent"]
#     ]  #
#     user_info: Optional[Dict[str, str]]  # 사용자 정보


# product_seller_recommendation: 제품 및 판매자 추천
# product_analysis: 사용자가 선택한 제품에 대한 분석 및 상세 정보
# negotiation: 판매자와의 협상 내용
#     product_seller_recommendation: Annotated[
#         Optional[str], "Product & Seller recommendation"
#     ]  # 제품 및 판매자 추천
#     product_analysis: Annotated[
#         Optional[str],
#         "Given the product selected by the user, analyze it and provide detailed information.",
#     ]  # 크롤링된 데이터
#     negotiation: Annotated[
#         Optional[str], "Negotiation with the seller"
#     ]  # 판매자와의 협상
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
