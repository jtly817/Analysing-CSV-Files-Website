import os
import pandas as pd
import glob

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'CSV_files')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class CSVService:
    def __init__(self, conn):
        self.conn = conn

    def save_csv(self, file):
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        df = pd.read_csv(file_path)

        table_name = os.path.splitext(filename)[0]
        df.to_sql(table_name, self.conn, index=False, if_exists='replace')
        return filename, table_name

    def delete_single_csv(self, table_name):
        cursor = self.conn.cursor()
        try:
            # Drop table
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

            # Remove the matching CSV file (if it exists)
            csv_path = os.path.join(UPLOAD_FOLDER, f"{table_name}.csv")
            os.remove(csv_path)
            return True, f"Deleted file: {table_name}.csv & Dropped table: {table_name}"
        except Exception as e:
            return False, f"Error deleting {table_name}: {e}"    

    def delete_all_csvs(self):
        deleted, dropped = [], []
        cursor = self.conn.cursor()

        for file in glob.glob(os.path.join(UPLOAD_FOLDER, "*.csv")):
            os.remove(file)
            table_name = os.path.splitext(os.path.basename(file))[0]
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            deleted.append(os.path.basename(file))
            dropped.append(table_name)

        self.conn.commit()
        return deleted, dropped
