import streamlit as st
import mysql.connector

# Function to create a connection to the MySQL database
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",
            port=3306,
            user="root",
            password="Str0ngP@ss!DB",
            database="aus_db"
        )
        return connection
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
        return None

# Function to fetch bowling data
def fetch_bowling_data():
    connection = create_connection()
    if connection is not None:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM bowl")  # Your query
        data = cursor.fetchall()
        cursor.close()
        connection.close()
        return data
    return []

# Main function for bowling page
def main():
    st.title("Bowling Statistics")
    bowling_data = fetch_bowling_data()
    
    # Display the data in a table
    if bowling_data:
        st.write("### Bowling Table")
        st.table(bowling_data)
    else:
        st.write("No data available.")
