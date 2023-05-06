import utils
from keras.utils import np_utils, to_categorical
import numpy as np
from keras.models import model_from_json
from keras import losses, models, optimizers
from keras.optimizers import Adam
import keras.backend as K

def train_model(model_weights, features, labels):
    
    lb = utils.get_label_encoder()
    y = np_utils.to_categorical(lb.fit_transform(labels))

    X = features

    # Load pre-trained model and weights
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(model_weights)
    loaded_model.compile(optimizer = Adam(learning_rate=0.05), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on new data
    loaded_model.fit(X, y, epochs=10, batch_size=32)
    loaded_model.save_weights('updated_weights.h5')


