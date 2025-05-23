import pandas as pd
from flask import session

class SQLService:
    def __init__(self, conn):
        self.conn = conn

    def run_query(self, sql_string):
        df = pd.read_sql_query(sql_string, self.conn)
        session['last_display'] = df.to_json()
        return df.to_html(index=False)

    def get_table_data(self, table_name):
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)
        session['last_display'] = df.to_json()
        return df.to_html(index=False)

    def download_last_display(self):
        df = pd.read_json(session['last_display'])
        return df.to_csv(index=False)

    def get_table_dataframe(self, table_name):
        query = f"SELECT * FROM {table_name}"
        return pd.read_sql_query(query, self.conn)