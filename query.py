from urllib.error import ContentTooShortError
import freesound
import json
import os
from db import AudioDatabase
from common import FEATURE_TAGS, FREESOUND_AUTH_PATH, SOUNDS_DIR

client = freesound.FreesoundClient()
with open(FREESOUND_AUTH_PATH) as f:
    auth_info = json.load(f)
client.set_token(auth_info["access_token"], auth_type="oauth")

page_size = 150
MAX_PAGE=2
extension = "wav"

db = AudioDatabase()

for tag in FEATURE_TAGS:
    for page in range(1, MAX_PAGE+1):
        print(f"{tag}: page {page}")
        sounds = client.text_search(query="", filter=f"duration:[0 TO 30] type:{extension} tag:{tag}", page=page, page_size=page_size, fields="id,tags", sort="downloads_desc")

        db.insert_sounds(sounds)

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

