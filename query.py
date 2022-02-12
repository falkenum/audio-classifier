import freesound
import json
from db import Database
from common import FREESOUND_AUTH_PATH

client = freesound.FreesoundClient()
with open(FREESOUND_AUTH_PATH) as f:
    auth_info = json.load(f)
client.set_token(auth_info["access_token"], auth_type="oauth")

page_size = 150
MAX_PAGE=100
extension = "wav"

db = Database()
descriptors = [
    "lowlevel.spectral_energy.max",
    "lowlevel.spectral_energy.mean",
    "lowlevel.spectral_energy.min",
]

descriptors_str = ",".join(descriptors)

for page in range(1, MAX_PAGE+1):
    sounds = client.text_search(query="", filter=f"type:{extension}", page=page, page_size=page_size, fields="id,tags,analysis", descriptors=descriptors_str, sort="downloads_desc")
    db.insert_sounds([sound for sound in sounds if sound.json_dict.get("analysis") is not None])

    print(f"page {page}")
