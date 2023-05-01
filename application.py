from flask import Flask, request
# import requests
# import predict_song_emotion

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get():
    return "Pod is running..."

@app.route('/v1/predict', methods = ['POST'])
def predict():
    song_url = request.form['song_url']
    model_weights_url = request.form['model_weights_url']
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

if __name__ == "__main__":
    app.run()
