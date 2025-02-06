import msgpack
import mediapipe as mp
import imageio as iio

from dotenv import load_dotenv
from flask import Flask, request, Response

from sys import exit
from os import environ
from pathlib import Path

load_dotenv()

words_dir = Path(environ.get("WORD_LIBRARY_LOCATION") or "words/").resolve()

if not words_dir.is_dir():
    words_dir.mkdir()

app = Flask(__name__)

mp_holistic = mp.solutions.holistic

def extract_results(landmarks):
    return [(res.x, res.y, res.z) for res in landmarks.landmark] if landmarks else None

def process_frames(cap):
    with mp_holistic.Holistic() as holistic:
        for data in cap:
            data.flags.writeable = False
            results = holistic.process(data)

            rh = extract_results(results.right_hand_landmarks)
            lh = extract_results(results.left_hand_landmarks)
            face = extract_results(results.face_landmarks)

            yield msgpack.packb((rh, lh, face))

@app.route("/api/word/<string:word>")
def rt_word(word: str):
    path = words_dir.joinpath(f"{word}.msgpack").resolve()
    if path.is_file():
        return Response(path.read_bytes(), mimetype="application/x-msgpack")
    else:
        return Response("Word not found", mimetype="text/plain"), 404

@app.route("/api/mark", methods=["POST"])
def rt_mark():
    stream = request.data
    cap = iio.imiter(stream, plugin="pyav", extension=".webm")
    frames = process_frames(cap)
    return Response(frames, mimetype="application/x-msgpack")

cors_allow = environ.get("CORS_ORIGIN_ALLOW")

@app.after_request
def after_req(resp):
    if cors_allow is not None: 
        headers = {
            "Access-Control-Allow-Origin": cors_allow,
            "Access-Control-Allow-Headers": "*"
        }
        resp.headers.update(headers)
    return resp

port = environ.get("FLASK_PORT") or 5000

def run_server():
    app.run(port=port, debug=(environ.get("FLASK_DEBUG") or "0") == "1")

if __name__ == "__main__":
    run_server()

