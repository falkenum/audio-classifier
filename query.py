from urllib.error import ContentTooShortError
import freesound
import os
import torchaudio
import torch
import pickle
import json

from common import AC_ANALYSIS_PATH, SAMPLE_RATE, SOUNDS_DIR, TAGS_PATH

client = freesound.FreesoundClient()
with open("freesound_auth.json") as f:
    auth_info = json.load(f)
client.set_token(auth_info["access_token"], auth_type="oauth")

# page_size = 15
MAX_PAGE=500
extension = "wav"
tags={}
ac_analysis={}
for page in range(1, MAX_PAGE+1):
    results = client.text_search(query="", filter=f"duration:[0 TO 30] type:{extension}", page=page, fields="id,name,tags,ac_analysis", sort="downloads_desc")

    for sound in results:
        filename = f"{sound.id}.{extension}"
        filepath = f"{SOUNDS_DIR}{filename}"
        # if not os.path.exists(filepath):
        #     while True:
        #         try:
        #             sound.retrieve(SOUNDSDIR, name=filename)
        #             break
        #         except (ContentTooShortError, OSError):
        #             print("got download error, trying again")

        #     waveform, samplerate = torchaudio.load(filepath)
        #     resampler = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=SAMPLERATE)
        #     waveform = resampler(waveform)
        #     if len(waveform) > 1:
        #         new_waveform = torch.zeros(1, waveform.shape[1])
        #         for i in range(waveform.shape[1]):
        #             new_waveform[0, i] = torch.mean(waveform[:, i])
        #         waveform = new_waveform
            
        #     torchaudio.save(filepath, waveform, SAMPLERATE)

        if sound.json_dict.get("ac_analysis") is None:
            continue

        tags[int(sound.id)] = sound.tags
        ac_analysis[int(sound.id)] = sound.json_dict.get("ac_analysis")

        print(f"{sound.id}: {sound.name}")


with open(TAGS_PATH, "wb") as f:
    pickle.dump(tags, f)

with open(AC_ANALYSIS_PATH, "wb") as f:
    pickle.dump(ac_analysis, f)
