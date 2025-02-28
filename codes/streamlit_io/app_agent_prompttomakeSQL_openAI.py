import streamlit as st
import os
import sqlite3
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
import traceback

# Page configuration
st.set_page_config(
    page_title="Chinook DB Natural Language Query",
    layout="wide"
)

# Initialize session state for API key
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None

# App title and description
col1, col2 = st.columns([3, 2])

with col1:
    st.title("Chinook DB Natural Language Query")
    st.markdown("""
    이 애플리케이션은 자연어를 사용하여 Chinook 데이터베이스에 질의할 수 있게 해주는 LLM 에이전트입니다.
    기술적 장벽 없이 일상 언어로 데이터베이스를 탐색해보세요.
    """)

with col2:
    st.image("https://www.sqlitetutorial.net/wp-content/uploads/2015/11/sqlite-sample-database-color.jpg", 
             caption="Chinook 데이터베이스 스키마", 
             use_column_width=True)

# API Key input
with st.sidebar:
    st.subheader("OpenAI API 키 설정")
    api_key = st.text_input("OpenAI API 키를 입력하세요:", 
                           value=st.session_state.openai_api_key,
                           type="password")
    
    if api_key:
        st.session_state.openai_api_key = api_key
        st.success("API 키가 설정되었습니다!")

# Main section
col_query, col_result = st.columns([1, 1])

# Sample queries
sample_queries = [
    "가장 많은 음악을 구매한 고객 10명을 보여주세요",
    "어떤 아티스트가 가장 많은 앨범을 가지고 있나요?",
    "장르별 트랙 수와 평균 가격을 알려주세요",
    "미국 고객들의 총 구매액은 얼마인가요?",
    "Rock 장르의 트랙 중 가장 긴 10개 트랙은 무엇인가요?"
]

with col_query:
    st.subheader("데이터베이스 질문")
    
    selected_query = st.selectbox(
        "예제 질문을 선택하거나 직접 질문하세요:",
        ["직접 입력하기"] + sample_queries
    )
    
    if selected_query == "직접 입력하기":
        user_query = st.text_area("자연어로 질문을 입력하세요:", height=150)
    else:
        user_query = st.text_area("자연어로 질문을 입력하세요:", value=selected_query, height=150)
    
    query_button = st.button("질문하기")

with col_result:
    st.subheader("처리 결과")
    result_container = st.empty()

# Function to query the database
def query_database(query_text, agent_executor):
    try:
        # Execute the query
        result = agent_executor.run(query_text)
        return result
    
    except Exception as e:
        error_msg = f"오류가 발생했습니다: {str(e)}"
        st.error(error_msg)
        return f"처리 중 오류가 발생했습니다: {str(e)}\n\n상세 오류: {traceback.format_exc()}"

# Function to initialize the agent executor
def set_query_agent_executor_database(api_key):
    try:
        # Check if API key is provided
        if not api_key:
            return None, "OpenAI API 키가 설정되지 않았습니다. 왼쪽 사이드바에서 API 키를 입력해주세요."
        
        # Connect to the SQLite database
        if not os.path.exists("chinook.db"):
            return None, "chinook.db 파일을 찾을 수 없습니다! 애플리케이션과 같은 디렉토리에 chinook.db 파일이 있는지 확인하세요."
        
        db = SQLDatabase.from_uri("sqlite:///chinook.db")
        
        # Create an OpenAI LLM with the provided API key
        llm = OpenAI(temperature=0, openai_api_key=api_key)
        
        # Create a SQL agent
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            top_k=10
        )
        return agent_executor, None
    except Exception as e:
        return None, f"에이전트 초기화 중 오류가 발생했습니다: {str(e)}"

# Check database and initialize agent
if st.session_state.openai_api_key and st.session_state.agent_executor is None:
    agent_executor, error_msg = set_query_agent_executor_database(st.session_state.openai_api_key)
    if error_msg:
        st.error(error_msg)
    else:
        st.session_state.agent_executor = agent_executor

# Process the query when button is clicked
if query_button and user_query:
    if not st.session_state.openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다. 왼쪽 사이드바에서 API 키를 입력해주세요.")
    elif st.session_state.agent_executor is None:
        # Try to initialize the agent again
        agent_executor, error_msg = set_query_agent_executor_database(st.session_state.openai_api_key)
        if error_msg:
            st.error(error_msg)
        else:
            st.session_state.agent_executor = agent_executor
            with st.spinner('질문을 처리 중입니다...'):
                try:
                    response = query_database(user_query, st.session_state.agent_executor)
                    result_container.markdown(f"""
                    ### 질문:
                    {user_query}
                    
                    ### 답변:
                    {response}
                    """)
                except Exception as e:
                    st.error(f"처리 중 오류가 발생했습니다: {str(e)}")
    else:
        with st.spinner('질문을 처리 중입니다...'):
            try:
                response = query_database(user_query, st.session_state.agent_executor)
                result_container.markdown(f"""
                ### 질문:
                {user_query}
                
                ### 답변:
                {response}
                """)
            except Exception as e:
                st.error(f"처리 중 오류가 발생했습니다: {str(e)}")

# Footer with cautions
st.markdown("---")
st.subheader("주의 사항")
st.info("""
1. 이 애플리케이션을 사용하기 위해서는 OpenAI API 키가 필요합니다. 왼쪽 사이드바에서 API 키를 입력하세요.
2. 자연어 처리는 100% 정확하지 않을 수 있으며, 복잡한 쿼리의 경우 결과가 예상과 다를 수 있습니다.
3. 데이터베이스 스키마를 참고하여 관련 테이블과 필드에 대한 질문을 작성하면 더 정확한 결과를 얻을 수 있습니다.
4. 처리 속도는 질문의 복잡성과 API 응답 시간에 따라 달라질 수 있습니다.
5. API 키는 사용자의 브라우저 세션에만 저장되며, 서버에 저장되지 않습니다.
""")