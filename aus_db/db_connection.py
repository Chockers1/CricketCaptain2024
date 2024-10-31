import mysql.connector

def get_db_connection():
    connection = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="Str0ngP@ss!DB",
        database="aus_db"
    )
    return connection
