from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages.chat import ChatMessage
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.tools import tool
from kiwipiepy.utils import Stopwords
from kiwipiepy import Kiwi
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
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_teddynote import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium.webdriver.support import expected_conditions as EC
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup as bs
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from dotenv import load_dotenv
from typing import List, Dict
import pandas as pd
import requests
import warnings
import time
import json
import re
import os

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=os.environ["OPENAI_API_KEY"]
)

# kiwi
kiwi = Kiwi(typos="basic", model_type="sbg")
stopwords = Stopwords()
stopwords.remove(("사람", "NNG"))


def kiwi_tokenize(text):
    text = "".join(text)
    result = kiwi.tokenize(text, stopwords=stopwords, normalize_coda=True)
    N_list = [i.form.lower() for i in result if i.tag in ["NNG", "NNP", "SL", "SN"]]
    return N_list


# 상품 정보 모델
class Topic(BaseModel):
    page_content: str = Field(description="제품에 대한 설명")
    metadata: dict = Field(
        description="product_name, category(예: 축구화, 풋살화),product_price, product_discount_price로 구성된 메타데이터"
    )


# 가격 파싱 함수
def parse_price(price_str):
    try:
        return int(price_str.replace(",", "").replace(" 원", "").strip())
    except (ValueError, AttributeError):
        return "정보없음"


# Tools 정의
@tool
def product_recommend(query: str) -> str:
    """구매자가 상품에 대한 추천 및 정보를 원할 시 이에 대한 정보를 제공하는 도구입니다."""
    tavily_tool = TavilySearch(
        include_domains=["naver.com", "youtube.com"],
        exclude_domains=["spam.com", "ads.com"],
    )
    result = tavily_tool.search(
        query=query,  # 검색 쿼리
        topic="general",  # 일반 주제
        max_results=10,  # 최대 10개 결과
        include_answer=True,  # 답변 포함
        include_raw_content=True,  # 원본 콘텐츠 포함
        format_output=True,  # 결과 포맷팅
    )
    return result


@tool
def site_search(query: str) -> str:
    """구매자가 원하는 물건을 팔고 있는 전문 온라인 판매점에 대한 정보를 제공할 때 사용하는 도구입니다."""
    tavily_tool = TavilySearch(
        include_domains=["naver.com"],
        exclude_domains=["spam.com", "ads.com"],
    )
    result = tavily_tool.search(
        query=query,  # 검색 쿼리
        topic="general",  # 일반 주제
        max_results=10,  # 최대 10개 결과
        include_answer=True,  # 답변 포함
        include_raw_content=True,  # 원본 콘텐츠 포함
        format_output=True,  # 결과 포맷팅
    )
    return result


@tool
def crawl_product_info(site_keyword: str, product_keyword: str):
    """
    특정 쇼핑몰에서 제품 키워드로 제품 정보들을 크롤링합니다.

    Args:
        site_keyword (str): 검색할 쇼핑몰 키워드 (예: '크레이지 11')
        product_keyword (str): 검색할 제품 키워드 (예: '나이키 머큐리얼')

    Returns:
        list[dict]: 제품 정보 딕셔너리들의 리스트
    """
    # 옵션 설정
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("window-size=1920x1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    )
    options.add_experimental_option(
        "excludeSwitches", ["enable-logging", "enable-automation"]
    )
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    wait = WebDriverWait(driver, 5)
    results = []

    try:
        # Google에서 쇼핑몰 검색
        driver.get("https://www.google.co.kr/")
        search_box = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "gLFyf"))
        )
        search_box.send_keys(site_keyword)
        search_box.send_keys(Keys.RETURN)

        # 첫 번째 결과 클릭
        first_result = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "LC20lb"))
        )
        first_result.click()

        # 쇼핑몰 내부에서 상품 검색
        search_input = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, "//input[@type='text' or @name='search' or @name='keyword']")
            )
        )
        search_input.send_keys(product_keyword)
        search_input.send_keys(Keys.RETURN)

        # 상품 링크 수집
        product_elements = wait.until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.itemInfo"))
        )
        product_urls = [el.get_attribute("href") for el in product_elements[:5]]

        # 각 상품 페이지 방문
        for url in product_urls:
            driver.get(url)

            product_information = driver.find_elements(
                By.CSS_SELECTOR, ".itemDetailPage-inner"
            )
            product_characteristic = driver.find_elements(
                By.CSS_SELECTOR, ".addInfo-mainContent"
            )
            product_images = driver.find_elements(
                By.CSS_SELECTOR, ".ls-is-cached.lazyloaded"
            )

            info_text = "\n".join(
                [el.text.strip() for el in product_information if el.text.strip()]
            )
            char_text = "\n".join(
                [el.text.strip() for el in product_characteristic if el.text.strip()]
            )
            image_urls = [
                img.get_attribute("src")
                for img in product_images
                if img.get_attribute("src")
            ]

            product_data = {
                "url": url,
                "information": info_text,
                "characteristic": char_text,
                "images": image_urls,
            }

            results.append(product_data)

    except Exception as e:
        print(f"[오류 발생] {e}")
    finally:
        driver.quit()

    return results


@tool
def analyze_and_recommend_products(results: List[Dict], user_query: str):
    """크롤링된 제품 정보(results)를 바탕으로 제품 요약 및 분석을 수행하고, 사용자 쿼리에 따라 적합한 제품들을 비교표 형태로 추천해주는 도구입니다.

    - 제품 정보 요약: 크롤링된 제품 정보를 LLM을 이용해 요약합니다.
    - 메타데이터 기반 검색: 요약된 정보에서 가격, 카테고리, 제품명을 기준으로 사용자의 쿼리에 맞는 제품을 필터링합니다.
    - 결과 출력: 사용자 질문에 부합하는 최대 5개의 제품을 비교표로 제공합니다.

    Args:
        results (List[Dict]): 크롤링된 원시 제품 데이터 리스트
        user_query (str): 사용자의 검색 조건 및 선호 기준을 포함한 자연어 질의

    Returns:
        str: 사용자 조건에 부합하는 제품 비교표 (Markdown 테이블 형태)
    """
    # 1. 요약용 파서 및 프롬프트 설정
    parser = JsonOutputParser(pydantic_object=Topic)
    summary_prompt = ChatPromptTemplate.from_template(
        """
        You are a product summary expert. Summarise the product information as best as possible.

        # Rules:
        - page_content에 제품의 가격과 사이즈를 꼭 포함하세요.
        - **하나의 제품이 아닌 information에 있는 각 제품에 대해 요약해주세요.**
        - **각 제품별로 정보를 요약하세요.**
        - product_price와 product_discounted_price을 잘 구분하세요.

        # Input:
        {information}

        # Format:
        {format_instructions}
        """
    ).partial(format_instructions=parser.get_format_instructions())

    # 2. 요약 실행
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = summary_prompt | model | parser
    summaries = chain.invoke(
        {"information": results}
    )  # results는 크롤링된 product list

    # 3. 문서화 및 가격 파싱
    docs = [
        Document(
            page_content=item["page_content"],
            metadata={
                "product_name": item["metadata"]["product_name"],
                "category": item["metadata"]["category"],
                "product_price": parse_price(item["metadata"]["product_price"]),
                "product_discounte_price": parse_price(
                    item["metadata"]["product_discount_price"]
                ),
            },
        )
        for item in summaries
    ]

    # 4. 벡터스토어 생성
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embedding)

    # 5. 메타데이터 필드 정보 정의
    metadata_field_info = [
        AttributeInfo(name="product_name", description="제품의 이름", type="string"),
        AttributeInfo(name="category", description="제품의 카테고리", type="string"),
        AttributeInfo(
            name="product_price",
            description="제품의 원래 가격 (할인 전)",
            type="integer",
        ),
        AttributeInfo(
            name="product_discounte_price",
            description="제품의 할인된 가격",
            type="integer",
        ),
    ]

    # 6. SelfQueryRetriever 설정
    retriever = SelfQueryRetriever.from_llm(
        llm=model,
        vectorstore=vectorstore,
        document_contents="Brief summary of a product",
        metadata_field_info=metadata_field_info,
        enable_limit=True,
        search_kwargs={"k": 2},
    )

    # 7. 최종 출력 프롬프트
    final_prompt = PromptTemplate.from_template(
        """
        당신은 사용자가 원하는 상품을 찾아주는 쇼핑 어드바이저입니다. 아래 규칙을 따라, 사용자 질문에 가장 적합한 상품들을 '판매처'별로 비교해 보여주세요.

        # 규칙 (Rules):
        - context에는 각 '판매처'에서 판매 중인 상품 목록이 포함되어 있습니다.
        - 사용자의 질문에서 원하는 **가격 범위**와 **주요 특징(기능)**에 맞는 상품만 선별하세요.
        - 상품은 **최대 5개까지** 제시해 주세요.
        - 각 상품은 다음 정보를 포함해 비교표 형식으로 출력하세요:
            - 판매처, 상품명, 가격, 특징
            - **특징은 구체적으로** 작성하세요.

        # 사용자 질문 (User Question):
        {question}

        # 참고 데이터 (Context):
        {context}

        # 답변 형식 (Answer Format):
        ** 제품 비교표**
        | 판매처     | 상품명            | 가격       | 특징                                                |
        |------------|-------------------|------------|-----------------------------------------------------|
        | **판매처1** | 상품명1           | ₩123,000   | ✔ 블루투스 5.1 지원, ✔ 5,000mAh 배터리, ✔ 무게 320g |
        | **판매처2** | 상품명2           | ₩198,000   | ✔ 15.6인치 FHD, ✔ HDMI/USB-C 지원 등               |
        """
    )

    final_chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | final_prompt
        | model
        | StrOutputParser()
    )

    # 8. 사용자 쿼리 실행
    return final_chain.invoke(user_query)


@tool
def negotiation(information: str) -> str:
    """
    Buyer Agent가 구매자의 협상 조건을 바탕으로 판매자와 가격, 배송, 기타 조건에 대해 최대 3회에 걸쳐 협상을 수행하는 도구입니다.

    - 입력된 information에는 구매자가 선택한 제품 및 판매처, 협상 조건(예: 희망 가격, 배송 조건 등)에 대한 요약 정보가 포함되어야 합니다.
    - 이 도구는 판매자 역할의 에이전트와 자연어 대화를 통해 가격 인하 또는 혜택 확보를 목표로 협상을 진행합니다.
    - 협상은 최대 3라운드로 진행되며, 마지막 라운드에서는 합의를 반드시 완료해야 합니다.
    - 출력은 판매자와의 협상 결과에 대한 한국어 자연어 응답입니다.

    Args:
        information (str): 구매자의 협상 조건 요약 정보 (예: 제품명, 희망 가격, 배송 요청 등)

    Returns:
        str: 판매자와의 협상 결과에 대한 자연어 응답 (한국어)
    """
    prompt = PromptTemplate.from_template(
        """
        ###
        당신은 구매자의 요약된 정보를 활용하여 판매자와 상호작용하면서 구매자를 대신하여 구매를 수행합니다.
        
        ###
        You are a Buyer Assistant tasked with negotiating with a Seller Assistant.
        Your goal is to secure the best possible deal for the buyer, such as price discounts or free shipping.
        You do not need to perfectly meet the Buyer's initial requirements but should aim for a reasonable compromise.
        Over three rounds of negotiations, you will interact with the Seller Assistant.
        By the end of the third round, you must agree to the seller's offer and conclude the negotiation.
        Communicate directly with the Seller Assistant and respond in Korean.
        
        # Information:
        {information}
        """
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = {"information": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return chain.invoke(information)
