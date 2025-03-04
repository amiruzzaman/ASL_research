import msgpack
import mediapipe as mp
import imageio as iio

from dotenv import load_dotenv
from flask import Flask, request, Response

from os import environ
from pathlib import Path

load_dotenv()

words_dir = Path(environ.get("WORD_LIBRARY_LOCATION") or "words/").resolve()

app = Flask(__name__)

mp_holistic = mp.solutions.holistic


def extract_results(landmarks):
    return (
        [{"x": res.x, "y": res.y, "z": res.z} for res in landmarks.landmark]
        if landmarks
        else None
    )


def make_relative_to(landmarks, reference, reference_point_idx, origin_idx):
    if reference and landmarks:
        origin_point = landmarks[origin_idx]
        reference_point = reference[reference_point_idx]

        def map_fn(global_point):
            return {
                "x": reference_point["x"] + (global_point["x"] - origin_point["x"]),
                "y": reference_point["y"] + (global_point["y"] - origin_point["y"]),
                "z": reference_point["z"] + (global_point["z"] - origin_point["z"]),
            }

        return list(map(map_fn, landmarks))
    else:
        return landmarks


def process_frames(cap, fps):
    yield msgpack.packb({"fps": round(fps)})
    with mp_holistic.Holistic(refine_face_landmarks=True) as holistic:
        for data in cap:
            data.flags.writeable = False
            holistic_results = holistic.process(data)
            pose = extract_results(holistic_results.pose_landmarks)
            face = make_relative_to(
                extract_results(holistic_results.face_landmarks), pose, 0, 0
            )
            rh = make_relative_to(
                extract_results(holistic_results.right_hand_landmarks), pose, 16, 0
            )
            lh = make_relative_to(
                extract_results(holistic_results.left_hand_landmarks), pose, 15, 0
            )
            yield msgpack.packb({"face": face, "pose": pose, "hands": (rh, lh)})


@app.route("/api/mark", methods=["POST"])
def rt_mark():
    if request.content_type == "video/webm":
        stream = request.data
        cap = iio.imiter(stream, plugin="pyav", extension=".webm")
        meta = iio.v3.immeta(stream, plugin="pyav", extension=".webm")
        fps = meta["fps"]
        frames = process_frames(cap, fps)
        return Response(frames, mimetype="application/x-msgpack")
    else:
        return Response("Expected `video/webm` video!"), 415


@app.route("/api/word/<string:word>", methods=["GET"])
def rt_word(word: str):
    path = words_dir.joinpath(f"{word}.msgpack").resolve()
    if path.is_file():
        return Response(path.read_bytes(), mimetype="application/x-msgpack")
    else:
        return Response("Word not found", mimetype="text/plain"), 404


@app.route("/api/word/<string:word>", methods=["POST"])
def rt_upload_word(word: str):
    if request.content_type == "application/x-msgpack":
        try:
            word_data = msgpack.unpackb(request.data)
            print(len(word_data[0]))
            if (
                type(word_data) is list
                and len(word_data) != 0
                and type(word_data[0]) is list
                and len(word_data[0]) == 4
            ):
                path = words_dir.joinpath(f"{word}.msgpack").resolve()
                path.write_bytes(msgpack.packb(word_data))
                return Response("Created"), 201
            else:
                return Response("Word data is not of the right shape"), 400
        except Exception as e:
            print(f"Invalid data received\n{e}")
            return Response("Invalid msgpack sent"), 400
    else:
        return Response("Data is not msgpack"), 415


cors_allow = environ.get("CORS_ORIGIN_ALLOW")


@app.after_request
def after_req(resp):
    if cors_allow is not None:
        headers = {
            "Access-Control-Allow-Origin": cors_allow,
            "Access-Control-Allow-Headers": "*",
        }
        resp.headers.update(headers)
    return resp


port = environ.get("FLASK_PORT") or 5000


def run_server():
    app.run(port=port, debug=(environ.get("FLASK_DEBUG") or "0") == "1")


if __name__ == "__main__":
    run_server()
