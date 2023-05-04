from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
from keras.optimizers import Adam
from utils import prepare_data

import numpy as np
import pandas as pd
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def get_feature_from_song_url (song_url):

    df = pd.DataFrame({'path': [song_url]})

    n = 30
    aug = 0

    X = prepare_data(df, n, aug)
    return X

def predict_emotion(model_weights, feature):

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(model_weights)
    loaded_model.compile(optimizer = Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    prediction = loaded_model.predict(feature)
    label = np.argmax(prediction)

    label_encoder = LabelEncoder()

    string_labels = ['ANGRY', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']
    label_encoder.fit(string_labels)

    string_label = label_encoder.inverse_transform([label])[0]

    return string_label


if __name__ == "__main__":

    model_weights_id = sys.argv[1]
    song_url = sys.argv[2]

    feature = get_feature_from_song_url(song_url)
    result = predict_emotion(model_weights_id, feature)
    print(result)
