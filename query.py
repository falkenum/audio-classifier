from urllib.error import ContentTooShortError
import freesound
import os
import torchaudio
import torch
import pickle
import json

from common import SAMPLERATE

client = freesound.FreesoundClient()
with open("freesound_auth.json") as f:
    auth_info = json.load(f)
client.set_token(auth_info["access_token"], auth_type="oauth")

# page_size = 15
MAX_PAGE=5
OUTDIR= "./sounds/"
extension = "wav"
tags={}
for page in range(1, MAX_PAGE+1):
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    results = client.text_search(query="", filter=f"duration:[0 TO 30] type:{extension}", page=page, fields="id,name,tags", sort="downloads_desc")

    for sound in results:
        filename = f"{sound.id}.{extension}"
        filepath = f"{OUTDIR}{filename}"
        if not os.path.exists(filepath):
            while True:
                try:
                    sound.retrieve(OUTDIR, name=filename)
                    break
                except (ContentTooShortError, OSError):
                    print("got download error, trying again")

            waveform, samplerate = torchaudio.load(filepath)
            resampler = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=SAMPLERATE)
            waveform = resampler(waveform)
            if len(waveform) > 1:
                new_waveform = torch.zeros(1, waveform.shape[1])
                for i in range(waveform.shape[1]):
                    new_waveform[0, i] = torch.mean(waveform[:, i])
                waveform = new_waveform
            
            torchaudio.save(filepath, waveform, SAMPLERATE)
            print(f"{sound.id}: {sound.name}")
            tags[int(sound.id)] = sound.tags


with open(f"./tags.pickle", "wb") as f:
    pickle.dump(tags, f)