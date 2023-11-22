import numpy as np
import soundfile as sf
import os
paths=["/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/dev-clean/","/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/test-clean/","/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/train-clean-360/"]
#Find out those file's from the librispeech dataset which have more than 15sec of the audio signal.


file_=[]
for path in paths:
    dataset=path.split("/")[-2]
    print(dataset)
    for f in os.listdir(path):
        for d in os.listdir(path+f):
            for fs in os.listdir(path+f+"/"+d):
                if ".flac" in fs:
                    file_length=sf.read(path+f+"/"+d+"/"+fs)[0].shape[0]
                    if file_length >= 240000:
                        file_.append((fs,dataset))

print(len(file_))
np.save("speech_files_15sec.npy",file_)



