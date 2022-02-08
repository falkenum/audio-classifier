import pickle
import torch
import torchaudio
import os
import threading
from common import AudioClassDataset, SOUNDSDIR, CLASSES_MAP_REV, FFT_SIZE, DATAPATH
dataset = AudioClassDataset()

if os.path.exists(DATAPATH):
    with open(DATAPATH, "rb") as f:
        dataset = pickle.load(f)

spectrogram = torchaudio.transforms.Spectrogram(n_fft=FFT_SIZE)
to_db = torchaudio.transforms.AmplitudeToDB()

lock = threading.Lock()
class ProcessThread(threading.Thread):
    def __init__(self, class_path) -> None:
        threading.Thread.__init__(self)
        self.class_path = class_path
    
    def run(self):
        lock.acquire()
        id_set = dataset.id_set.copy()
        lock.release()
        for file_path in self.class_path.iterdir():
            if file_path.stem not in id_set:
                self.process_file(file_path)

    def process_file(self, file_path):
        waveform, samplerate = torchaudio.load(file_path)
        spec = spectrogram(waveform)
        spec_db = to_db(spec)[0]
        label = CLASSES_MAP_REV[self.class_path.name]
        id = int(file_path.stem)
        # return spec_db, label, id
        lock.acquire()
        dataset.add_samples(spec_db, label, id)
        print("added", self.class_path.name, file_path.name)
        lock.release()

threads = []
for class_path in SOUNDSDIR.iterdir():
    threads.append(ProcessThread(class_path))
    threads[-1].start()

for thread in threads:
    thread.join()

with open(DATAPATH, "wb") as f:
    pickle.dump(dataset, f)