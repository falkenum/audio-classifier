import psycopg2
import pandas as pd

dbparams = {
    "host":"localhost",
    "database":"audio",
    "user":"postgres",
    "password":"postgres",
}

class AudioDatabase:
    def __init__(self) -> None:
        self.conn = psycopg2.connect(**dbparams)
        self
    
    def __del__(self):
        self.conn.close()
    
    def insert_sounds(self, sounds):
        cur = self.conn.cursor()
        values = []
        for sound in sounds:
            id = sound.id
            tags = sound.tags
            value = cur.mogrify("(%s, %s)", (id, tags)).decode("utf-8")
            values.append(value)
        values_str = ",".join(values)
        query = f'INSERT INTO sounds (id, tags) VALUES {values_str} ON CONFLICT (id) DO NOTHING'
        cur.execute(query)

        self.conn.commit()
    
    def get_sounds(self, limit):
        query = f"SELECT * FROM sounds ORDER BY id ASC LIMIT {limit}"
        return pd.read_sql(query, self.conn)
    
    def insert_sample(self, source_id, spec_idx, data, label):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO samples (source_id, spec_idx, data, label) VALUES (%s, %s, %s, %s) ON CONFLICT (source_id, spec_idx) DO NOTHING", (source_id, spec_idx, data, label))

        self.conn.commit()



# db = Database()
# db.insert_sounds([])