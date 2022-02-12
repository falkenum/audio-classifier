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
        values = []
        for sound in sounds:
            id = sound.id
            tags = sound.tags
            se_mean = float(sound.analysis.json_dict["lowlevel"]["spectral_energy"]["mean"])
            se_max = float(sound.analysis.json_dict["lowlevel"]["spectral_energy"]["max"])
            se_min = float(sound.analysis.json_dict["lowlevel"]["spectral_energy"]["min"])
            value = cur.mogrify("(%s, %s, %s, %s, %s)", (id, tags, se_mean, se_max, se_min)).decode("utf-8")
            values.append(value)
        values_str = ",".join(values)
        query = f'INSERT INTO "sounds" (id, tags, se_mean, se_max, se_min) VALUES {values_str} ON CONFLICT (id) DO NOTHING'
        cur.execute(query)

        self.conn.commit()
    
    def get_sounds(self):
        cur = self.conn.cursor()
        cur.execute("select * from sounds")

        return cur.fetchall()



# db = Database()
# db.insert_sounds([])