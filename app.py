from flask import Flask, render_template, request, url_for, jsonify
import lib

app = Flask(__name__)


def error(message):
    return jsonify({"error": message}), 400


@app.route("/cropVideo", methods=["POST", "GET"])
def crop_video():
    input_json = request.json
    gcs_vid_path = input_json["drivingVideoPath"]
    top_left = tuple([int(x) for x in input_json["topLeft"]])
    square_length = int(input_json["squareLength"])
    assert len(top_left) == 2
    result = lib.crop_video(gcs_vid_path, top_left, square_length)
    return jsonify(result)


@app.route("/generateResult", methods=["POST", "GET"])
def generate_result():
    input_json = request.json
    print(input_json)
    cropped_driving_vid_path = input_json["croppedDrivingVideoPath"]
    src_img_path = input_json["sourceImagePath"]
    result = lib.generate_result(cropped_driving_vid_path, src_img_path)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
