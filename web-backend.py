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

frontend_dir = Path(environ.get("FRONTEND_BUNDLE_LOCATION") or "web-frontend/dist/").resolve()

if not frontend_dir.is_dir():
    print("Error: FRONTEND_BUNDLE_LOCATION is not set to a valid directory built by `npm run build`.")
    exit(1)

def load_page_html(name: str):
    path: Path = frontend_dir.joinpath(name)
    if path.is_file():
        return path.read_text()
    else:
        print(f"Error: Expected a file named {name} within FRONTEND_BUNDLE_LOCATION")
        exit(1)

index_html = load_page_html("index.html")
e2a_html = load_page_html("e2a/index.html")
record_html = load_page_html("record/index.html")

app = Flask(__name__, static_url_path="/_astro", static_folder=frontend_dir.joinpath("_astro"))

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

@app.route("/word/<string:word>")
def rt_word(word: str):
    path = words_dir.joinpath(f"{word}.msgpack").resolve()
    if path.is_file():
        return Response(path.read_bytes(), mimetype="application/x-msgpack")
    else:
        return Response("Word not found", mimetype="text/plain"), 404

@app.route("/mark", methods=["POST"])
def rt_mark():
    stream = request.data
    cap = iio.imiter(stream, plugin="pyav", extension=".webm")
    frames = process_frames(cap)
    return Response(frames, mimetype="application/x-msgpack")

@app.route("/")
def rt_index():
    return Response(index_html, mimetype="text/html")

@app.route("/e2a")
def rt_e2a():
    return Response(e2a_html, mimetype="text/html")

@app.route("/record")
def rt_record():
    return Response(record_html, mimetype="text/html")

if __name__ == "__main__":
    app.run(debug=(environ.get("DEBUG") or "0") == "1")

