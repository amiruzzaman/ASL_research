import sys
import msgpack
import imageio as iio
from dotenv import load_dotenv
from flask import Flask, request, Response
import mediapipe as mp
import torch
from mediapipe.tasks.python import vision

from os import environ
from pathlib import Path

from ml.api.english_to_gloss import EnglishToGloss
from ml.api.asl_to_english import ASLToEnglish

HolisticLandmarker = vision.HolisticLandmarker

english_to_gloss = EnglishToGloss()
asl_to_english = ASLToEnglish()

load_dotenv()

model_path = environ.get("HOLISTIC_MODEL_PATH")

if model_path is None or model_path == "":
    print("ERROR: Model path for Holistic is needed (set HOLISTIC_MODEL_PATH)")
    sys.exit(1)

words_dir = Path(environ.get("WORD_LIBRARY_LOCATION") or "words/").resolve()

app = Flask(__name__)

holistic_options = vision.HolisticLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
)


def extract_results(landmarks):
    return (
        [{"x": res.x, "y": res.y, "z": res.z} for res in landmarks]
        if landmarks and len(landmarks) != 0
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
    frame = 0
    yield msgpack.packb({"fps": round(fps)})
    with HolisticLandmarker.create_from_options(holistic_options) as holistic:
        for data in cap:
            data.flags.writeable = False
            frame_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=data)
            holistic_results = holistic.detect_for_video(frame_img, frame)
            frame += 1
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


VIDEO_TYPES = {"video/webm": ".webm", "video/mp4": ".mp4"}


@app.route("/api/mark", methods=["POST"])
def rt_mark():
    ext = VIDEO_TYPES.get(request.content_type)
    if ext is not None:
        stream = request.data
        cap = iio.imiter(stream, plugin="pyav", extension=ext)
        meta = iio.v3.immeta(stream, plugin="pyav", extension=ext)
        fps = meta["fps"]
        frames = process_frames(cap, fps)
        return Response(frames, mimetype="application/x-msgpack")
    else:
        return Response("Expected WEBM or MP4 video!"), 415


@app.route("/api/gloss", methods=["POST"])
def rt_gloss():
    sentence = request.data.decode()
    words = english_to_gloss.translate(sentence)
    return Response(msgpack.packb(words), mimetype="application/x-msgpack")


@app.route("/api/a2e", methods=["POST"])
def rt_a2e():
    ext = VIDEO_TYPES.get(request.content_type)
    if ext is not None:
        stream = request.data
        cap = iio.imiter(stream, plugin="pyav", extension=ext)
        sequence = []
        buf = []
        for frame in cap:
            buf.append(frame)
            if len(buf) == 30:
                id, word = asl_to_english.translate_sign(torch.tensor(buf))
                sequence.append(buf)
                buf = []

        words = asl_to_english.translate(torch.tensor(sequence))
        return Response(msgpack.packb(words), mimetype="application/x-msgpack")
    else:
        return Response("Expected WEBM or MP4 video!"), 415


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
            if (
                type(word_data) is dict
                and word_data.get("fps") is not None
                and word_data.get("frames") is not None
            ):
                fps = word_data["fps"]
                if type(fps) is not int:
                    return Response("FPS information is not an integer"), 400
                elif fps < 0:
                    return Response("FPS must be a positive integer"), 400
                frame_data = word_data["frames"]
                if (
                    type(frame_data) is list
                    and len(frame_data) != 0
                    and type(frame_data[0]) is dict
                ):
                    if not words_dir.is_dir():
                        words_dir.mkdir(parents=True, exist_ok=True)
                    path = words_dir.joinpath(f"{word}.msgpack").resolve()
                    path.write_bytes(msgpack.packb(word_data))
                    print(f'New word "{word}" saved to {path}')
                    return Response("Created"), 201
                else:
                    return Response("Frame data is not a list of dictionaries"), 400
            else:
                return Response(
                    "Word data is not a dict or is missing 'fps' and 'frames'"
                ), 400
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
