from flask import Flask, request, send_file
import requests
import predict_song_emotion
import utils
import training

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get():
    return "Pod is running..."

@app.route('/v1/predict', methods = ['POST'])
def predict():
    data = request.get_json()
    song_url = data['song_url']
    model_weights_url = data['model_weights_url']
    song_response = requests.get(song_url)
    model_weights_response = requests.get(model_weights_url)

    song_data = song_response.content
    model_weights_data = model_weights_response.content
    
    with open('model_weights.h5', 'wb') as f:
        f.write(model_weights_data)

    with open('songs.mp3', 'wb') as f:
        f.write(song_data)

    song_url = 'songs.mp3'
    model_weights = 'model_weights.h5'

    feature = predict_song_emotion.get_feature_from_song_url(song_url)
    emotion = predict_song_emotion.predict_emotion(model_weights, feature)
    
    return emotion

@app.route('/v1/re-train', methods = ['POST'])
def re_train():
    data = request.get_json()
    model_weights_url = data['model_weights_url']
    song_url_list = data['song_urls']
    emotion_list = data['emotions']

    model_weights_response = requests.get(model_weights_url)
    model_weights_data = model_weights_response.content
    
    with open('model_weights.h5', 'wb') as f:
        f.write(model_weights_data)

    for i in range(len(song_url_list)):
        song_url_response = requests.get(song_url_list[i])
        song_data = song_url_response.content
        with open('song' + str(i) + '.mp3', 'wb') as f:
            f.write(song_data)
    
    df = utils.create_df(emotion_list)
    n = 30
    aug = 0

    # X for features
    X = utils.prepare_data(df, n, aug)
    
    # y for labels
    y = df['label']

    training.train_model("model_weights.h5", X, y)

    return send_file('updated_weights.h5', as_attachment=True)


if __name__ == "__main__":
    app.run()
