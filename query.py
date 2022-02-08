from urllib.error import ContentTooShortError
import freesound
from os import environ

client = freesound.FreesoundClient()
api_token = environ.get("FREESOUND_TOKEN")
client.set_token(api_token, auth_type="oauth")
 
# results = client.text_search(query="violin")
query="bass"
results = client.text_search(query=query, filter='duration:[0 TO 30] type:wav',)

for sound in results:
    # sound.retrieve_preview(".",sound.name+".mp3")
    while True:
        try:
            sound.retrieve(f"./sounds/{query}/", name=f"{sound.id}.wav")
            print(sound.name)
            break
        except ContentTooShortError:
            print("got download error, trying again")
