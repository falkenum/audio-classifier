from urllib.error import ContentTooShortError
import freesound
import os

client = freesound.FreesoundClient()
api_token = os.environ.get("FREESOUND_TOKEN")
client.set_token(api_token, auth_type="oauth")

# results = client.text_search(query="violin")
tags = ["bass", "percussion", "guitar", "trumpet", "violin"]
page = 2
extension = "wav"

for tag in tags:
    outdir = f"./sounds/{tag}/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    results = client.text_search(query=tag, filter=f"duration:[0 TO 30] type:{extension} tag:{tag}", page=page)

    for sound in results:
        while True:
            try:
                filename = f"{sound.id}.{extension}"
                filepath = f"{outdir}{filename}"
                if not os.path.exists(filepath):
                    sound.retrieve(outdir, name=filename)
                    print(sound.name)
                break
            except (ContentTooShortError, OSError):
                print("got download error, trying again")

