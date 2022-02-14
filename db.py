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
        return pd.read_sql(query, self.conn).to_records(index=False)

    def get_sound_ids_from_samples(self, limit, shuffle):
        if shuffle:
            query = f"SELECT * FROM (SELECT DISTINCT source_id FROM samples LIMIT {limit}) AS ids ORDER BY RANDOM ()"
        else:
            query = f"SELECT DISTINCT source_id FROM samples ORDER BY source_id ASC LIMIT {limit}"

        return map(lambda elt: elt[0], pd.read_sql(query, self.conn).to_records(index=False))

    def get_num_samples(self):
        query = "SELECT COUNT(source_id) FROM samples"
        return pd.read_sql(query, self.conn).to_records(index=False)[0][0]

    def insert_sample(self, source_id, spec_idx, data, label):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO samples (source_id, spec_idx, data, label) VALUES (%s, %s, %s, %s) ON CONFLICT (source_id, spec_idx) DO NOTHING", (source_id, spec_idx, data, label))

        self.conn.commit()
    
    def get_samples_for_id(self, source_id):
        return pd.read_sql(f"SELECT data, label FROM samples WHERE source_id={source_id} ORDER BY spec_idx ASC", self.conn).to_records(index=False)


# db = Database()
# db.insert_sounds([])