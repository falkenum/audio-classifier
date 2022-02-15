from urllib.error import ContentTooShortError
import freesound
import json
import os
from db import AudioDatabase
from common import FREESOUND_AUTH_PATH, SOUNDS_DIR

client = freesound.FreesoundClient()
with open(FREESOUND_AUTH_PATH) as f:
    auth_info = json.load(f)
client.set_token(auth_info["access_token"], auth_type="oauth")

page_size = 150
MAX_PAGE=2
extension = "wav"

db = AudioDatabase()

# query_tags = ["guitar", "piano", "violin", "trumpet"]
query_tags = []
for tag in query_tags:
    for page in range(1, MAX_PAGE+1):
        print(f"{tag}: page {page}")
        sounds = client.text_search(query="", filter=f"duration:[3 TO 20] type:{extension} tag:{tag}", page=page, page_size=page_size, fields="id,tags", sort="downloads_desc")

        for sound in sounds:
            filename = f"{sound.id}.{extension}"
            filepath = f"{SOUNDS_DIR}/{filename}"
            if not os.path.exists(filepath):
                while True:
                    try:
                        sound.retrieve(SOUNDS_DIR, name=filename)
                        break
                    except ContentTooShortError:
                        print("download failed, trying again")

        db.insert_sounds(sounds)
