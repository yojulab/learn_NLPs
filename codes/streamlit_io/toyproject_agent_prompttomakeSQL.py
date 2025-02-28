import sqlite3
# 1. 데이터베이스 연결
# conn = sqlite3.connect("codes/streamlit_io/chinook.db")
# cursor = conn.cursor()

# 쿼리 실행 및 결과 출력 (예: 고객 정보 조회)
# cursor.execute("SELECT CustomerId, FirstName, LastName, Country FROM customers LIMIT 10")

# # # 결과 가져오기 및 출력
# print("고객 정보 (ID, 이름, 성, 국가):")
# for row in cursor.fetchall():
#     print(f"ID: {row[0]}, 이름: {row[1]}, 성: {row[2]}, 국가: {row[3]}")

# 2. Langchain에서 사용할 데이터베이스 객체 생성
from langchain.sql_database import SQLDatabase

# SQLDatabase 객체를 생성하여 연결
db = SQLDatabase.from_uri("sqlite:///codes/streamlit_io/chinook.db")

# 테이블 목록 확인
# tables = db.get_table_names()
# print("데이터베이스 테이블 목록:")
# for table in tables:
#     print(f"- {table}")

# # 고객 정보 조회
# query = "SELECT CustomerId, FirstName, LastName, Country FROM customers LIMIT 5"
# result = db.run(query)
# print("\n고객 정보:")
# print(result)

# # 앨범 정보 조회
# query = "SELECT AlbumId, Title FROM albums LIMIT 5"
# result = db.run(query)
# print("\n앨범 정보:")
# print(result)

import os
os. environ[ "OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")


from langchain.agents import AgentType, create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI  # langchain_openai 패키지에서 임포트
from langchain.utilities import SQLDatabase  # 또는 langchain_community.utilities

llm = ChatOpenAI (model_name="gpt-4-1106-preview", temperature=0)
toolkit = SQLDatabaseToolkit (db=db, llm=llm)

few_shots = {
  "List all artists.": "SELECT * FROM artists;"
  ,"Find all albums for the artist 'AC/DC'.": "SELECT albums.Title FROM albums JOIN artists ON albums.ArtistId = artists.ArtistId WHERE artists.Name = 'AC/DC'"
  ,"List all tracks in the 'Rock' genre.": "SELECT tracks.Name FROM tracks JOIN genres ON tracks.GenreId = genres.GenreId WHERE genres.Name = 'Rock'"
  ,"Find the total duration of all tracks.": "SELECT SUM(Mi |liseconds) FROM tracks;"
  ,"List all customers from Canada.": "SELECT * FROM customers WHERE Country = 'Canada' ; "
  ,"How many tracks are there in the album with ID ?": "SELECT COUNT(*) FROM tracks WHERE AlbumId = 5; "
  ,"Find the total number of invoices.": "SELECT COUNT(*) FROM invoices; "
  ,"List all tracks that are longer than 5 minutes.": "SELECT * FROM tracks WHERE Mi | liseconds > 300000;"
  ,"Who are the top 5 customers by total purchase?": "SELECT c.FirstName, c.LastName, SUM(i.Total) as TotalPurchase FROM customers c JOIN invoices i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId ORDER BY TotalPurchase DESC "
  ,"Which albums are from the year 2000?": "SELECT * FROM albums WHERE strft ime('%Y', ReleaseDate) = '2000' ;"
  ,"How many employees are there": 'SELECT COUNT (*) FROM "employee"' ,
}

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
embeddings = OpenAIEmbeddings ( )
few_shot_docs = [
  Document (page_content=question, metadata={"sal _query": few_shots[question] })
  for question in few_shots.keys()
]
vector_db = FAISS. from_documents(few_shot_docs, embeddings)
retriever = vector_db.as_retriever ()

from langchain.agents.agent_toolkits import create_retriever_tool
tool_description = """
이 도구는 유사한 예시를 이해하여 사용자 질문에 적용하는 데 도움이 됩니다.
이 도구에 입력하는 내용은 사용자 질문이어야 합니다.
"""
retriever_tool = create_retriever_tool (
    retriever, name="sql_get_similar_examples" , description=tool_description
)
custom_tool_list = [retriever_tool]

# 기존 agent 코드
custom_suffix = """
먼저 제가 알고 있는 비슷한 예제를 가져와야 합니다.
예제가 쿼리를 구성하기에 충분하다면 쿼리를 작성할 수 있습니다.
그렇지 않으면 데이터베이스의 테이블을 살펴보고 쿼리할 수 있는 항목을 확인할 수 있습니다.
그런 다음 가장 관련성이 높은 테이블의 스키마를 쿼리해야 합니다.
"""

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    extra_tools=custom_tool_list, 
    suffix=custom_suffix,
)

from langchain.callbacks.base import BaseCallbackHandler

class MessageCaptureCallback(BaseCallbackHandler):
    def __init__(self):
        self.messages = []
    
    def on_agent_action(self, action, **kwargs):
        self.messages.append({"agent_action": str(action)})
    
    def on_tool_end(self, output, **kwargs):
        self.messages.append({"tool_output": output})
    
    def on_chain_end(self, outputs, **kwargs):
        self.messages.append({"chain_output": str(outputs)})


message_callback = MessageCaptureCallback()

# 콜백 핸들러를 사용하여 agent 실행
result = agent.run("How many employees do we have?", callbacks=[message_callback])

# 수집된 모든 메시지 출력
from pprint import pprint

# 모든 메시지 확인
print("\n==== 수집된 모든 메시지 ====\n")
for i, msg in enumerate(message_callback.messages):
    print(f"메시지 {i+1} : {msg.keys()}):")
    # pprint(msg, width=100, depth=2)
    print("-" * 50)

# 결과와 메시지를 변수에 저장
agent_execution_data = {
    "final_result": result,
    "all_messages": message_callback.messages
}

# 특정 유형의 메시지만 필터링하기 (예: agent_action만 보기)
agent_actions_only = [msg for msg in message_callback.messages if "agent_action" in msg.keys()]
print("\n==== Agent 액션만 필터링 ====\n")
pprint(agent_actions_only)