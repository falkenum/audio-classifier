from urllib.error import ContentTooShortError
import freesound
import os
import torchaudio
import torch

from common import CLASSES_LIST, SAMPLERATE

client = freesound.FreesoundClient()
api_token = os.environ.get("FREESOUND_TOKEN")
client.set_token(api_token, auth_type="oauth")

page = 2
extension = "wav"

for tag in CLASSES_LIST:
    outdir = f"./sounds/{tag}/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    results = client.text_search(query=tag, filter=f"duration:[0 TO 30] type:{extension} tag:{tag}", page=page)

    for sound in results:
        filename = f"{sound.id}.{extension}"
        filepath = f"{outdir}{filename}"
        while True:
            try:
                if not os.path.exists(filepath):
                    sound.retrieve(outdir, name=filename)

                    print(sound.name)
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

