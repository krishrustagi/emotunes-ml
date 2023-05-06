import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
from sklearn.preprocessing import LabelEncoder

sampling_rate=44100
audio_duration=2.5
n_mfcc = 30

'''
1. Data Augmentation method   
'''
def speedNpitch(data):
    """
    Speed and Pitch Tuning.
    """
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.2  / length_change # try changing 1.0 to 2.0 ... =D
    tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data

'''
2. Extracting the MFCC feature as an image (Matrix format).  
'''
def prepare_data(df, n, aug):
    X = np.empty(shape=(df.shape[0], n, 216, 1))
    input_length = sampling_rate * audio_duration
    
    cnt = 0
    for fname in tqdm(df.path):
        file_path = fname
        data, _ = librosa.load(file_path, sr=sampling_rate
                               ,res_type="kaiser_fast"
                               ,duration=2.5
                               ,offset=0.5
                              )

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

        # Augmentation? 
        if aug == 1:
            data = speedNpitch(data)
        
        # MFCC extraction 
        MFCC = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)
        MFCC = np.expand_dims(MFCC, axis=-1)
        X[cnt,] = MFCC

        cnt += 1
    
    return X

def create_df(emotion_list):
    data = {
        'label': emotion_list, 
        'path': ["song" + str(i) + ".mp3" for i in range(len(emotion_list))]
    }

    df = pd.DataFrame(data, columns=['label', 'path'])

    new_data = {
        'label': 'ANGRY',
        'path': 'default_songs/anger.mp3'
    }

    df = df.append(new_data, ignore_index = True)
    
    new_data = {
        'label': 'FEAR',
        'path': 'default_songs/fear.mp3'
    }

    df = df.append(new_data, ignore_index = True)
    
    new_data = {
        'label': 'HAPPY',
        'path': 'default_songs/happy.mp3'
    }

    df = df.append(new_data, ignore_index = True)
    
    new_data = {
        'label': 'NEUTRAL',
        'path': 'default_songs/neutral.mp3'
    }

    df = df.append(new_data, ignore_index = True)
    
    new_data = {
        'label': 'SAD',
        'path': 'default_songs/sad.mp3'
    }

    df = df.append(new_data, ignore_index = True)

    new_data = {
        'label': 'SURPRISE',
        'path': 'default_songs/surprise.mp3'
    }

    df = df.append(new_data, ignore_index = True)
    print(df)

    return df

def get_label_encoder():
    
    label_encoder = LabelEncoder()

    string_labels = ['ANGRY', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']
    label_encoder.fit(string_labels)

    return label_encoder