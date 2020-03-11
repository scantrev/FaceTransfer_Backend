import os
import shutil
import time

import firebase_admin
import imageio
import IPython
import skimage
from firebase_admin import credentials, storage
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
    # Crop video
    print("Cropping video")
    x, y = top_left

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
    # Now do processing


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
    clip.write_videofile(vid_path, codec="libx264")


if __name__ == "__main__":
    # crop_video("AAE8F8E3-CD86-4B98-88D9-EAB5A35AAE22/drivingVideo.mov", [0, 279], 720)
    generate_result(
        "89AF5A3A-6EBC-4331-BB8F-6CDA06364F27/sourceImage.jpg",
        "89AF5A3A-6EBC-4331-BB8F-6CDA06364F27/sourceImage.jpg",
    )
