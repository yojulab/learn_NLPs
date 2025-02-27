import streamlit as st
import pandas as pd
import sqlite3
import os

def main():
    # Set page title and layout
    st.set_page_config(
        page_title="Chinook 데이터베이스 질의 도우미",
        layout="wide"
    )
    
    # Header section with app description
    st.title("Chinook SQL 데이터베이스 질의 도우미")
    
    # Two columns for description and DB image
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.write("""
        ## 기능 설명
        이 애플리케이션은 Chinook 데이터베이스 관련 질문을 입력하면 SQL 쿼리를 생성하고 결과를 보여줍니다.
        Chinook SQLite 데이터베이스는 음악 스토어의 데이터를 담고 있으며, 오른쪽에 보이는 스키마를 참고하여 
        질문을 작성해주세요.
        
        데이터베이스에는 고객, 아티스트, 앨범, 트랙, 인보이스 등의 정보가 포함되어 있습니다.
        """)
    
    with col2:
        # Display DB schema image
        st.image("https://www.sqlitetutorial.net/wp-content/uploads/2015/11/sqlite-sample-database-color.jpg", 
                 caption="Chinook 데이터베이스 스키마", 
                 use_column_width=True)
    
    st.divider()
    
    # Main content - Question input area
    st.subheader("데이터베이스 질문 입력")
    
    # Check if a db file exists
    if not os.path.exists('chinook.db'):
        st.error("Error: chinook.db 파일을 찾을 수 없습니다. 애플리케이션 폴더에 chinook.db 파일이 있는지 확인해주세요.")
        return
    
    # Text area for question input
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ""
        
    question = st.text_area(
        "데이터베이스에 대한 질문을 입력하세요:",
        value=st.session_state.question_input,
        height=150,
        key="question_area",
        help="데이터베이스 구조를 참고하여 질문을 작성해주세요."
    )
    
    # Process button
    if st.button("질문 처리", type="primary"):
        if question:
            st.session_state.last_question = question
            process_question(question)
        else:
            st.warning("질문을 입력해주세요.")
    
    st.divider()
    
    # Example questions section
    st.subheader("예제 질문")
    
    # Example questions with corresponding SQL queries
    examples = [
        "모든 고객의 이름과 이메일을 보여주세요.",
        "앨범 수가 가장 많은 아티스트 상위 5명을 보여주세요.",
        "각 장르별 트랙 수를 내림차순으로 정렬해서 보여주세요.",
        "각 국가별 고객 수를 내림차순으로 정렬해서 보여주세요.",
        "가장 많은 인보이스를 가진 고객은 누구인가요?"
    ]
    
    # Create columns for example buttons
    cols = st.columns(3)
    for i, example in enumerate(examples):
        col_index = i % 3
        with cols[col_index]:
            if st.button(f"예제 {i+1}", key=f"example_{i}"):
                st.session_state.question_input = example
                st.experimental_rerun()
    
    # Display result section if a question was processed
    if 'last_question' in st.session_state and 'query_result' in st.session_state:
        st.divider()
        st.subheader("결과")
        st.write(f"**질문:** {st.session_state.last_question}")
        
        st.write("**생성된 SQL 쿼리:**")
        st.code(st.session_state.generated_query, language="sql")
        
        st.write("**결과 데이터:**")
        if isinstance(st.session_state.query_result, pd.DataFrame) and not st.session_state.query_result.empty:
            st.dataframe(st.session_state.query_result)
        elif 'error_message' in st.session_state:
            st.error(st.session_state.error_message)
        else:
            st.info("결과가 없습니다.")

def process_question(question):
    """
    Process the database question, generate SQL query and execute it against chinook.db
    """
    try:
        # Connect to the chinook database
        conn = sqlite3.connect('chinook.db')
        
        # Generate query based on the question
        query = generate_query_from_question(question)
        
        # Store the generated query in session state
        st.session_state.generated_query = query
        
        try:
            # Execute the query and get results
            result = pd.read_sql_query(query, conn)
            st.session_state.query_result = result
        except Exception as e:
            # Handle SQL execution errors
            st.session_state.error_message = f"SQL 실행 오류: {str(e)}"
            st.session_state.query_result = pd.DataFrame()
        
        # Close the connection
        conn.close()
        
    except Exception as e:
        st.session_state.error_message = f"오류가 발생했습니다: {str(e)}"
        st.session_state.query_result = pd.DataFrame()

def generate_query_from_question(question):
    """
    Generate an SQL query based on the user's question.
    In a production environment, this could be replaced with a more sophisticated NLP model.
    """
    question = question.lower()
    
    if "고객" in question and "이름" in question and "이메일" in question:
        return """
        SELECT FirstName, LastName, Email 
        FROM customers 
        ORDER BY LastName;
        """
    
    elif "아티스트" in question and "앨범" in question and "상위" in question:
        return """
        SELECT artists.Name as ArtistName, COUNT(albums.AlbumId) as AlbumCount
        FROM artists
        JOIN albums ON artists.ArtistId = albums.ArtistId
        GROUP BY artists.ArtistId
        ORDER BY AlbumCount DESC
        LIMIT 5;
        """
    
    elif "장르" in question and "트랙" in question:
        return """
        SELECT genres.Name as Genre, COUNT(tracks.TrackId) as TrackCount
        FROM genres
        JOIN tracks ON genres.GenreId = tracks.GenreId
        GROUP BY genres.GenreId
        ORDER BY TrackCount DESC;
        """
    
    elif "국가" in question and "고객" in question:
        return """
        SELECT Country, COUNT(*) as CustomerCount
        FROM customers
        GROUP BY Country
        ORDER BY CustomerCount DESC;
        """
    
    elif "인보이스" in question and "고객" in question:
        return """
        SELECT c.FirstName, c.LastName, COUNT(i.InvoiceId) as InvoiceCount
        FROM customers c
        JOIN invoices i ON c.CustomerId = i.CustomerId
        GROUP BY c.CustomerId
        ORDER BY InvoiceCount DESC
        LIMIT 1;
        """
    
    else:
        # Default query for unrecognized questions
        return """
        -- 질문에 맞는 쿼리를 생성할 수 없습니다.
        -- 아래는 데이터베이스 구조를 보여주는 쿼리입니다.
        SELECT name FROM sqlite_master WHERE type='table';
        """

if __name__ == "__main__":
    main()