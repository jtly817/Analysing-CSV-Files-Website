import pandas as pd
import os
from flask import session

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'CSV_files')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class TableService:
    def __init__(self, conn):
        self.conn = conn

    def get_all_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]

    def save_displayed_table(self, table_name):
        if not table_name: 
            return False, f"No name provided to save table."
        if table_name and 'last_display' not in session:
            return False, "No displayed table to save with name."
        try:
            df = pd.read_json(session['last_display'])

            # Save to SQLite
            df.to_sql(table_name, self.conn, index=False, if_exists='replace')

            # Save to CSV
            csv_path = os.path.join(UPLOAD_FOLDER, f"{table_name}.csv")
            df.to_csv(csv_path, index=False)

            return True, f"Saved current table as <b>{table_name}</b> (in DB and as CSV)."
        except Exception as e:
            return False, f"Error saving table: {e}"
