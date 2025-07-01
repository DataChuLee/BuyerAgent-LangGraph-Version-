import streamlit as st
import warnings
import sqlite3
import os
from langchain_core.messages.chat import ChatMessage
from dotenv import load_dotenv
from Workflow.workflow import create_workflow, stream_graph
from langchain_teddynote import logging
from langchain_core.messages import HumanMessage, AIMessage
from langchain_teddynote.messages import random_uuid

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# env 파일에서 OPENAI API KEY 들여옴
load_dotenv()

# LangChain 추적 시작
logging.langsmith("0701_BuyerAgent")

db_path = os.path.join("User_DB", "user_info.sqlite3")

# DB 연결 및 테이블 초기화
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS user_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT NOT NULL,
    user_type TEXT NOT NULL,
    thread_id TEXT NOT NULL UNIQUE
)
"""
)
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,
    message TEXT NOT NULL
)
"""
)
conn.commit()


# 사용자 정보 저장 함수
def save_user_info_to_db(user_name: str, user_type: str, thread_id: str):
    cursor.execute(
        """
    INSERT OR IGNORE INTO user_info (user_name, user_type, thread_id)
    VALUES (?, ?, ?)
    """,
        (user_name, user_type, thread_id),
    )
    conn.commit()


# 사용자별 메시지 저장 함수
def save_message_to_db(thread_id: str, role: str, message: str):
    cursor.execute(
        """
    INSERT INTO messages (thread_id, role, message)
    VALUES (?, ?, ?)
    """,
        (thread_id, role, message),
    )
    conn.commit()


st.set_page_config(
    page_title="Buyer Agent",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("검색부터 협상까지, AI가 자동으로 수행하는 쇼핑")
st.markdown("더 이상 여러 사이트를 비교하거나 가격을 흥정할 필요 없습니다.")
st.markdown("Buyer Agent는")
st.markdown("● 구매자가 원하는 조건을 파악하고")
st.markdown("● 제품과 판매처 정보를 자동으로 수집하며")
st.markdown("● 필요 시 판매자와의 협상도 진행합니다.")
st.markdown("쇼핑에 드는 시간과 스트레스를 줄이고,더 나은 조건의 구매를 실현하세요.")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "graph" not in st.session_state:
    st.session_state["graph"] = create_workflow()
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = random_uuid()
if "user_information" not in st.session_state:
    st.session_state["user_information"] = []

# 사이드바
with st.sidebar:
    st.header("사용자 정보 💡")
    user_type = st.selectbox("사용자 유형", ["구매자", "판매자"], index=0)
    user_name = st.text_input(f"{user_type} 이름", value="", key="user_name_input")

    if st.button("정보 설정", key="save_user"):
        if user_name:
            thread_id = f"{'buyer' if user_type == '구매자' else 'seller'}_{user_name}"
            st.session_state["thread_id"] = thread_id
            st.session_state["user_information"].append(
                {"user_name": user_name, "user_type": user_type}
            )
            save_user_info_to_db(user_name, user_type, thread_id)
            st.success(f"✅ {user_name}님({user_type})의 정보가 DB에 저장되었습니다.")

    st.markdown("---\n**등록된 사용자 정보**")
    if st.session_state["user_information"]:
        for idx, user_info in enumerate(st.session_state["user_information"]):
            st.write(f"👤 {user_info['user_name']} ({user_info['user_type']})")
    else:
        st.info("아직 등록된 정보가 없습니다.")

    st.markdown("---\n**대화 초기화**")
    if st.button("새로운 질문", type="primary"):
        st.session_state["messages"] = []
        st.session_state["thread_id"] = random_uuid()


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        if chat_message.role == "user":
            st.chat_message(chat_message.role, avatar="🙎‍♂️").write(chat_message.content)
        else:
            st.chat_message(chat_message.role, avatar="😊").write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))
    save_message_to_db(st.session_state["thread_id"], role, message)


def get_message_history():
    ret = []
    for chat_message in st.session_state["messages"]:
        if chat_message.role == "user":
            ret.append(HumanMessage(content=chat_message.content))
        else:
            ret.append(AIMessage(content=chat_message.content))

    return ret


# 이전 대화 기록 출력
print_messages()

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 만약에 사용자 입력이 들어오면...
if user_input:

    # 세션 상태에서 그래프 객체를 가져옴
    graph = st.session_state["graph"]

    if graph is not None:
        # 사용자 입력을 화면에 표시
        st.chat_message("user", avatar="🙎‍♂️").write(user_input)

        # AI 답변을 화면에 표시
        with st.chat_message("assistant", avatar="😊"):
            streamlit_container = st.empty()
            # 그래프를 호출하여 응답 생성
            response = stream_graph(
                graph,
                user_input,
                streamlit_container,
                thread_id=st.session_state["thread_id"],
            )

            # 응답에서 AI 답변 추출
            ai_answer = response["generation"]
            st.write(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
