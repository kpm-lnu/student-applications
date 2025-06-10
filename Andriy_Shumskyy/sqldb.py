import psycopg2
from dotenv import load_dotenv
import os
load_dotenv()



conn = psycopg2.connect(
    dbname=os.getenv("DBNAME"),
    user=os.getenv("USER"),
    password=os.getenv("PASSWORD"),
    host=os.getenv("HOST"),
    port=os.getenv("PORT")
)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS license_plates (
        id SERIAL PRIMARY KEY,
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        license_plate VARCHAR(20)
    )
''')

conn.commit()
cursor.close()
conn.close()

