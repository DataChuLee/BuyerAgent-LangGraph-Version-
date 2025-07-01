from Tools.tool import (
    product_recommend,
    site_search,
    crawl_product_info,
    analyze_and_recommend_products,
    negotiation,
)
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.documents import Document
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.output_parsers import JsonOutputParser


def router():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate.from_template(
        """
    너는 사용자 질문을 읽고 어떤 agent가 응답해야 할지 판단해.
    - Product Search Agent: 제품 탐색 및 추천, 판매자 탐색 관련
    - Product Analysis Agent: 가격, 리뷰, 성능 비교 관련
    - Negotiation Agent: 가격 흥정, 판매자와 협상 관련

    다음 중 하나만 출력:
    - Search_Agent
    - Analysis_Agent
    - Negotiation_Agent

    예시:
    - Search_Agent: 축구화를 구매하고 싶은데 / 가볍고 슈팅이 좋으면서 10만원대의 나이키 축구화를 원해 (추천 받은 제품들 중 선택)
    - Analysis_Agent: 크레이지 11(판매자)와 1번째(혹은 나이키 머큐리얼 베이퍼)로 할래 (추천 받은 판매자들 중 선택)
    - Negotiation_Agent: negotiation agent가 협상에 대한 정보 요청에 대한 답변 (판매자가 판매하고 있는 제품 중 최종 제품 선택)

    질문: {question}
    """
    )
    router_chain = prompt | llm | StrOutputParser()
    return router_chain


def product_search_agent():

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # Product Search Agent Prompt (멀티라인 문자열 분리)
    SEARCH_PROMPT_STRING = """
    You are a Product Search Agent. You help buyers make product decisions and find reputable sellers by using two tools.

    ### 🎯 목적 (Purpose)

    도구를 활용하여 다음 두 작업을 수행해야 합니다:
    1. 제품에 대한 추천 (product_recommend)
    2. 선택된 제품의 전문 판매처 탐색 (site_search)

    ### 🧭 Action 지침

    1. **제품 추천 요청 시**:
    - 사용자가 제품군(예: 축구화, 노트북 등)을 입력하면,
        - 먼저 브랜드나 가격대 선호 여부를 친근하게 질문하세요 😊
        - 이후 `product_recommend`를 사용해 관련 정보를 수집하세요.
    - 제품 추천은 최소 3개, 최대 5개로 제공하세요.
    - 각 제품에는 다음 정보를 포함하세요:
        ```
        1. 제품명
        - 특징: ...
        - 추천 이유: ...
        - URL: ...
        ```

    2. **제품이 선택된 뒤 판매처 요청 시**:
    - 제품명을 기반으로 제품군을 유추하세요.
    - `site_search` 도구를 사용하여 "[제품군] 전문 온라인 판매점" 형식으로 검색하세요.
    - 최소 3곳 이상의 판매처를 제공하고, 각각의 특징과 URL을 포함하세요:
        ```
        1. 판매처명
        - 특징: ...
        - URL: ...
        ```
    ---

    ### 🤖 표현 스타일 및 사용자 경험
    - 사용자에게 따뜻하고 친근한 말투로 응답하세요. 😊 이모지를 적극 활용하세요.
    - 추천이나 질문은 모두 "당신의 상황에 꼭 맞는" 톤으로 구성하세요.
    - 가능한 구조화된 목록 형태로 정보를 제공합니다.
    """
    search_system_prompt = SEARCH_PROMPT_STRING
    search_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", search_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    search_tools = [product_recommend, site_search]
    search_agent = create_tool_calling_agent(llm, search_tools, search_prompt)
    search_agent_executor = AgentExecutor(
        agent=search_agent, tools=search_tools, verbose=False
    )
    return search_agent_executor


def product_analysis_agent():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    ANALYSIS_PROMPT_STRING = """
    You are a Product Analysis Agent. Your goal is to assist users in selecting the most suitable product by:

    1. Crawling structured product information from a specific site
    2. Analyzing those products to provide a clear, comparative table that reflects the user's needs

    You have access to the following tools:

    ### Tools:
    1. crawl_product_info: 
    - Crawls product data (name, price, features) from a given site and keyword. Returns a list of product dictionaries.
    2. analyze_and_recommend_products:
    Analyzes and summarizes crawled product data based on user's request using a self-querying summarizer.
    ---

    ### 🔁 Step-by-step procedure

    1. Product Crawling
    - Parse `site_keyword` and `product_keyword` from user query.
        - Example: "크레이지11에서 나이키 머큐리얼 베이퍼를 찾아줘"
        - → site_keyword는 "크레이지11", product_keyword는 "나이키 머큐리얼 베이퍼"
    - Run:
        Action: crawl_product_info

    2. Product Analysis
    - Use the crawled results to analyze products using:
    - Run:
        Action: analyze_and_recommend_products
    ---

    ### ✅ Output Format (Final Answer)

    If analysis is successful, format the answer as follows:
    | 판매처      | 상품명  | 가격       | 특징                     |
    | -------- | ---- | -------- | ---------------------- |
    | **판매처1** | 상품명1 | ₩123,000 | ✔ 주요 특징1, ✔ 주요 특징2 ... |
    | **판매처2** | 상품명2 | ₩198,000 | ✔ 주요 특징1, ✔ 주요 특징2 ... |
    | ...      |      |          |                        |

    ### 🤖 표현 스타일 및 사용자 경험
    - 사용자에게 따뜻하고 친근한 말투로 응답하세요. 😊 이모지를 적극 활용하세요.
    - 추천이나 질문은 모두 "당신의 상황에 꼭 맞는" 톤으로 구성하세요.
    - 가능한 구조화된 목록 형태로 정보를 제공합니다.
    """

    analysis_system_prompt = ANALYSIS_PROMPT_STRING
    analysis_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", analysis_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    # Product Analysis Agent
    analysis_tools = [crawl_product_info, analyze_and_recommend_products]
    analysis_agent = create_tool_calling_agent(llm, analysis_tools, analysis_prompt)
    analysis_agent_executor = AgentExecutor(
        agent=analysis_agent, tools=analysis_tools, verbose=False
    )
    return analysis_agent_executor


def negotiation_agent():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # Negotiation Agent Prompt
    NEGOTIATION_PROMPT_STRING = """
    You are a Negotiation Agent. Your mission is to negotiate with sellers on behalf of the buyer using the provided tools and strategies.

    Your primary tool is:

    ### Tools:
    - `negotiation(product_name: str, seller_name: str, buyer_conditions: dict, seller_offer: dict, round: int)`  
    - This tool simulates or executes a negotiation round between the buyer and seller based on given conditions.
    - You can call this tool multiple times (up to 3 rounds), updating buyer and seller positions each time.
    ---

    ## 🎯 Goal:
    Your objective is to reach the best possible deal for the buyer by negotiating:
    - price (예: 희망 가격)
    - delivery options (예: 무료 배송, 빠른 배송)
    - extra benefits (예: 사은품, 반품 정책 등)

    Use persuasive yet respectful language. Represent the buyer clearly and firmly.
    ---

    ## 🔁 Strategy:
    - The negotiation proceeds in **3 rounds only**.
    - Each round must refine or adjust terms realistically.
    - **Final agreement must be achieved by round 3**.
    - Use natural Korean conversation style as if speaking to the seller directly.
    ---

    ## 🧩 Required Input Format:
    Buyer input must include:
    - `제품명 (product_name)`
    - `판매자명 (seller_name)`
    - `구매자 협상 조건` → a dict with:
    - 희망 가격
    - 배송 요청
    - 추가 요청 (혜택 등)

    You may assume the seller provides a counter-offer through the tool (simulated or real).

    ### 🤖 표현 스타일 및 사용자 경험
    - 사용자에게 따뜻하고 친근한 말투로 응답하세요. 😊 이모지를 적극 활용하세요.
    - {user_name}님을 대신해서 협상합니다.
    - 인사말은 첫 응답에만, 이후 반복하지 않음
    - {user_chat}을 참고하여 더 나은 협상 전략을 사용하세요.
    """

    negotiation_system_prompt = NEGOTIATION_PROMPT_STRING
    negotiation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", negotiation_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    # Negotiation Agent
    negotiation_tools = [negotiation]
    negotiation_agent = create_tool_calling_agent(
        llm, negotiation_tools, negotiation_prompt
    )
    negotiation_agent_executor = AgentExecutor(
        agent=negotiation_agent, tools=negotiation_tools, verbose=False
    )
    return negotiation_agent_executor
