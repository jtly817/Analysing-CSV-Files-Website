import sqlite3
import os

DATABASE_PATH = os.path.join('static', 'data.db')

def get_connection():
    if not os.path.exists(DATABASE_PATH):
        conn = sqlite3.connect(DATABASE_PATH)
        conn.close()
    return sqlite3.connect(DATABASE_PATH, check_same_thread=False)
