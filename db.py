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
    
    def get_bird_sounds(self, limit, shuffle=False):
        if shuffle:
            query = f"SELECT * FROM birds ORDER BY RANDOM () LIMIT {limit}"
        else:
            # TODO
            assert(False)
        return pd.read_sql(query, self.conn).to_records(index=False)

    def get_notes_sounds(self, limit, shuffle=False):
        if shuffle:
            query = f"SELECT * FROM notes ORDER BY RANDOM () LIMIT {limit}"
        else:
            # TODO
            assert(False)
        return pd.read_sql(query, self.conn).to_records(index=False)

    def get_catdog_sounds(self, limit, shuffle=False):
        if shuffle:
            query = f"SELECT * FROM catdog ORDER BY RANDOM () LIMIT {limit}"
        else:
            # TODO
            assert(False)
        return pd.read_sql(query, self.conn).to_records(index=False)

    def get_num_birds(self):
        cur = self.conn.cursor()
        query = f"SELECT COUNT(*) FROM (SELECT DISTINCT ebird_code FROM birds) AS unique_birds"
        cur.execute(query)
        return cur.fetchone()[0]

    def get_num_note_types(self):
        cur = self.conn.cursor()
        query = f"SELECT COUNT(*) FROM (SELECT DISTINCT type FROM notes) AS unique_notes"
        cur.execute(query)
        return cur.fetchone()[0]

# db = Database()
# db.insert_sounds([])