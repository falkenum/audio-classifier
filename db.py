import psycopg2
import pandas as pd

dbparams = {
    "host":"localhost",
    "database":"audio",
    "user":"postgres",
    "password":"postgres",
}

class Database:
    def __init__(self) -> None:
        self.conn = psycopg2.connect(**dbparams)
        self
    
    def __del__(self):
        self.conn.close()
    
    def insert_sounds(self, sounds):
        cur = self.conn.cursor()
        values = ",".join([cur.mogrify("(%s, %s, %s)", (sound.id, sound.name, sound.tags)).decode("utf-8") for sound in sounds])
        # print(values)
        query = f'INSERT INTO "sounds" (id, name, tags) VALUES {values} ON CONFLICT (id) DO NOTHING'
        # query = "SELECT table_name from information_schema.tables where table_schema = 'public'"
        cur.execute(query)
        # print(cur.fetchall())

        self.conn.commit()

# db = Database()
# db.insert_sounds([])