import os
import shutil
import time

import firebase_admin
import imageio
import IPython
import moviepy.editor as mp
import numpy as np
import skimage
import torch
from firebase_admin import credentials, storage
from moviepy import *
from moviepy.editor import *
from skimage.transform import resize
from tqdm import tqdm

import demo
import face_alignment

cred = credentials.Certificate("admin.json")
firebase_admin.initialize_app(cred, {"storageBucket": "facetransfer-bb67b.appspot.com"})

bucket = storage.bucket()

# Methods
def crop_video(gcs_vid_path, top_left, square_length):
    vid_path = gcs_vid_path
    job_id = gcs_vid_path.split("/")[0]
    if not os.path.exists(job_id):
        os.makedirs(job_id)
    download_from_gcs(gcs_vid_path, vid_path)
    clip = load_vid(vid_path)
    clip = fix(clip)
    # Crop video
    print("Cropping video")
    x, y = top_left

    # IPython.embed()
    def crop(get_frame, t):
        frame = get_frame(t)
        cropped_frame = frame[
            int(y) : int(y) + square_length, int(x) : int(x) + square_length
        ]
        resized_frame = resize(cropped_frame, (256, 256), preserve_range=True)
        return resized_frame

    cropped_clip = clip.fl(crop)
    cropped_path = "croppedDrivingVideo.mov"
    gcs_cropped_vid_path = cropped_vid_path = os.path.join(job_id, cropped_path)
    save_vid(cropped_vid_path, cropped_clip)
    upload_to_gcs(gcs_cropped_vid_path, cropped_vid_path)
    # shutil.rmtree(job_id)
    return {"croppedDrivingVideoPath": gcs_cropped_vid_path}


def generate_result(gcs_cropped_vid_path, gcs_src_img_path):
    job_id = gcs_cropped_vid_path.split("/")[0]
    cropped_vid_path = gcs_cropped_vid_path
    src_img_path = gcs_src_img_path
    if not os.path.exists(job_id):
        os.makedirs(job_id)
    download_from_gcs(gcs_cropped_vid_path, cropped_vid_path)
    download_from_gcs(gcs_src_img_path, src_img_path)
    clip = load_vid(cropped_vid_path)
    img = load_img(src_img_path)
    global source
    source = resize(img, (256, 256))
    print(source.shape)
    # print(source[0])
    # Prepare model
    print("Loading model")
    generator, kp_detector = demo.load_checkpoints(
        config_path="config/vox-256.yaml", checkpoint_path="vox-cpk.pth.tar",
    )
    print("Model loaded")
    global fps
    fps, duration = clip.fps, clip.duration
    print("FPS={},duration={}".format(fps, duration))
    num_frames = int(round(fps * duration))
    print(num_frames)
    driving_video = (
        np.asarray([clip.get_frame(i / clip.fps) for i in range(num_frames)]) / 255
    )
    print(driving_video[0].max(), driving_video[0].min())
    print(driving_video[0].shape)
    print("Driving video num frames={}".format(len(driving_video)))
    print("Animating")
    global predictions
    predictions = demo.make_animation(
        source, driving_video, generator, kp_detector, relative=True
    )
    print("Num predictions={}".format(len(predictions)))

    global frame_idx
    frame_idx = 0

    def get_animated_frame(get_frame, t):
        global predictions, frame_idx, source
        frame = predictions[frame_idx] * 255
        driving_frame = driving_video[frame_idx] * 255
        if frame_idx < len(predictions) - 1:
            frame_idx += 1
        combined_frame = np.hstack((driving_frame, frame))
        # print(combined_frame.shape)
        return combined_frame

    animated_clip = clip.fl(get_animated_frame)
    # IPython.embed()
    text_clip = TextClip(
        "FaceTransfer",
        font="Arial",
        fontsize=50,
        bg_color="black",
        color="white",
        align="center",
        method="caption",
        size=(512, 256),
    ).set_duration(1)
    final_clip = mp.concatenate_videoclips([animated_clip, text_clip])
    # final_clip = animated_clip
    gcs_gen_combined_vid_path = gen_combined_vid_path = os.path.join(
        job_id, "generatedCombinedVideo.mov"
    )
    save_vid(gen_combined_vid_path, final_clip)
    upload_to_gcs(gen_combined_vid_path, gcs_gen_combined_vid_path)
    return {"generatedCombinedVideoPath": gcs_gen_combined_vid_path}


# Utilities


def download_from_gcs(gcs_path, media_path):
    print("Downloading from {} to {}".format(gcs_path, media_path))
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(media_path)


def upload_to_gcs(gcs_path, media_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(media_path)


def load_vid(vid_path):
    print("Loading video from {}".format(vid_path))
    return VideoFileClip(vid_path)


def save_vid(vid_path, clip):
    print("Saving video to {}".format(vid_path))
    clip.write_videofile(vid_path, threads=4, codec="libx264", logger=None)
    print("Video saved")


def load_img(img_path):
    return imageio.imread(img_path)


def fix(video):
    if video.rotation == 90:
        video = video.resize(video.size[::-1])
        video.rotation = 0
    return video


if __name__ == "__main__":
    # crop_video("AAE8F8E3-CD86-4B98-88D9-EAB5A35AAE22/drivingVideo.mov", [0, 279], 720)
    generate_result(
        "89AF5A3A-6EBC-4331-BB8F-6CDA06364F27/croppedDrivingVideo.mov",
        "89AF5A3A-6EBC-4331-BB8F-6CDA06364F27/sourceImage.jpg",
    )
