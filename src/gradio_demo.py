import torch, uuid
import os, shutil
import string
import random
import tempfile
import json
from pydub import AudioSegment
from rq import get_current_job
import logging as log

from src.utils.b64 import base64_to_image, base64_to_path, path_to_base64
from src.utils.email_sender import EmailSender
from src.utils.blob_storage import BlobStorage
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from src.utils.speech_utils import get_speech_start_sec, get_audio_len_sec


def mp3_to_wav(mp3_filename, wav_filename, frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename, format="wav")


def gen_random_suffix(length=4):
    letters_and_digits = string.ascii_lowercase + string.digits
    suffix = "".join(random.choice(letters_and_digits) for _ in range(length))
    return suffix


class SadTalker:

    def __init__(
        self, checkpoint_path="checkpoints", config_path="src/config", lazy_load=False
    ):

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.device = device

        os.environ["TORCH_HOME"] = checkpoint_path

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path

    def test(
        self,
        source_img_b64,
        audio_b64_str,
        preprocess_type="crop",
        is_still_mode=False,
        enhancer=True,
        batch_size=1,
        size=256,
        pose_style=0,
        exp_scale=1.0,
        head_motion_scale=1.0,
        bg_img_b64=None,
        ref_blink_b64=None,
        ref_pose_b64=None,
    ):
        job = get_current_job()
        job_id = job.id

        source_image = base64_to_image(source_img_b64)
        bg_image = base64_to_image(bg_img_b64)
        ref_blink = base64_to_path(ref_blink_b64, ".mp4")
        ref_pose = base64_to_path(ref_pose_b64, ".mp4")
        audio_path = base64_to_path(audio_b64_str, ".wav")

        result_dir = "./results"

        video_path = f"{result_dir}/{job_id}.mp4"

        cmd = f'python inference.py --driven_audio {audio_path} --source_image {source_image} --result_dir {result_dir}{f" --bg_image {bg_image}" if bg_image else ""}{f" --ref_eyeblink {ref_blink}" if ref_blink else ""}{f" --ref_pose {ref_pose}" if ref_pose else ""} --final_vid_name {job_id}.mp4{" --still" if is_still_mode else ""} --preprocess {preprocess_type} --expression_scale {exp_scale} --head_motion_scale {head_motion_scale} --batch_size {batch_size} --size {size}{f" --enhancer gfpgan" if enhancer else ""}'
        os.system(cmd)
        log.info(f"Video generated and saved to {video_path}")

        return path_to_base64(video_path)

    def generate_avatar(
        self,
        source_img_b64,
        bg_img_b64,
        ref_blink_b64,
        ref_pose_b64,
        preprocess_type,
        is_still_mode,
        exp_scale,
        head_motion_scale,
        email,
        avatar_name,
    ):
        job = get_current_job()
        job_id = job.id

        job.meta["job_id"] = job_id
        job.meta["email"] = email
        job.meta["avatar_name"] = avatar_name
        job.meta["preprocess_type"] = preprocess_type
        job.meta["is_still_mode"] = is_still_mode
        job.meta["exp_scale"] = exp_scale
        job.meta["head_motion_scale"] = head_motion_scale
        job.save_meta()

        blob_storage = BlobStorage()

        source_image = base64_to_image(source_img_b64)
        bg_image = base64_to_image(bg_img_b64)
        ref_blink = base64_to_path(ref_blink_b64, ".mp4")
        ref_pose = base64_to_path(ref_pose_b64, ".mp4")

        log.info(f"Source image path: {source_image}")
        log.info(f"Background image path: {bg_image}")

        audio_dir = "./examples/custom/audio"
        result_dir = f"./examples/custom/result/{job_id}"

        if blob_storage.check_dir_exists(avatar_name):
            orig_avatar_name = avatar_name
            avatar_name = f"{avatar_name}-{gen_random_suffix()}"
            log.warning(
                f"Avatar with name: {orig_avatar_name} already exists, changing name to: {avatar_name}"
            )

        batch_size = int(os.environ.get("BATCH_SIZE", "2"))

        os.makedirs(result_dir, exist_ok=True)

        for audio in os.listdir(audio_dir):
            audio_path = os.path.join(audio_dir, audio)
            audio_name = os.path.splitext(audio)[0]
            video_path = f"{result_dir}/{audio_name}.mp4"

            log.info(f"Generating video for audio {audio}")
            cmd = f'python inference.py --driven_audio {audio_path} --source_image {source_image} --result_dir {result_dir}{f" --bg_image {bg_image}" if bg_image else ""}{f" --ref_eyeblink {ref_blink}" if ref_blink else ""}{f" --ref_pose {ref_pose}" if ref_pose else ""} --final_vid_name {audio_name}.mp4{" --still" if is_still_mode else ""} --preprocess {preprocess_type} --expression_scale {exp_scale} --head_motion_scale {head_motion_scale} --batch_size {batch_size} --size 512 --enhancer gfpgan'
            os.system(cmd)
            log.info(f"Video generated and saved to {video_path}")

            blob_storage.upload_file(video_path, avatar_name)
            log.info(f"Video {video_path} uploaded to storage")

        log.info(f"Generating default video")
        default_vid_name = "default-video.mp4"
        video_path = f"{result_dir}/{default_vid_name}"
        cmd = f'python inference.py --source_image {source_image} --result_dir {result_dir}{f" --bg_image {bg_image}" if bg_image else ""}{f" --ref_eyeblink {ref_blink}" if ref_blink else ""}{f" --ref_pose {ref_pose}" if ref_pose else ""} --final_vid_name {default_vid_name}{" --still" if is_still_mode else ""} --preprocess {preprocess_type} --expression_scale {exp_scale} --head_motion_scale {head_motion_scale} --batch_size {batch_size} --size 512 --enhancer gfpgan --idlemode --len 20'
        os.system(cmd)
        log.info(
            f"Default video generated and saved to {result_dir}/{default_vid_name}"
        )
        blob_storage.upload_file(video_path, avatar_name)
        log.info(f"Video {video_path} uploaded to storage")

        dh_face_base_uri = os.environ.get(
            "DH_FACE_BASE_URI", "https://customer-test.borndigital.ai/digital-human/"
        )
        dh_face_url = f"{dh_face_base_uri}?name={avatar_name}"

        email_sender = EmailSender()
        subject = "Link to your avatar!"
        html_body = f"""
                <html>
                <body>
                    <h1>Hello there,</h1>
                    <p>Your avatar is ready! 🎉</p>
                    <p>Click the link below to try it out:</p>
                    <a href="{dh_face_url}" target="_blank">Try Avatar</a>
                    <p>Enjoy your new avatar! 🤖</p>
                </body>
                </html>
                """
        email_sender.send(subject, html_body, [email])

        return dh_face_url

    def gen_custom_videos(
        self,
        source_img_b64,
        bg_img_b64,
        ref_blink_b64,
        ref_pose_b64,
        audios,
        preprocess_type,
        is_still_mode,
        exp_scale,
        head_motion_scale,
        avatar_name,
    ):
        job = get_current_job()
        job_id = job.id

        job.meta["job_id"] = job_id
        job.meta["avatar_name"] = avatar_name
        job.meta["preprocess_type"] = preprocess_type
        job.meta["is_still_mode"] = is_still_mode
        job.meta["exp_scale"] = exp_scale
        job.meta["head_motion_scale"] = head_motion_scale
        job.save_meta()

        blob_storage = BlobStorage()

        source_image = base64_to_image(source_img_b64)
        bg_image = base64_to_image(bg_img_b64)
        ref_blink = base64_to_path(ref_blink_b64, ".mp4")
        ref_pose = base64_to_path(ref_pose_b64, ".mp4")

        log.info(f"Source image path: {source_image}")
        log.info(f"Background image path: {bg_image}")

        result_dir = f"./examples/custom/result/{job_id}"
        os.makedirs(result_dir, exist_ok=True)

        batch_size = int(os.environ.get("BATCH_SIZE", "2"))

        config_filename = f"{avatar_name}_config.json"
        blob_name = f"{avatar_name}/{config_filename}"
        if blob_storage.check_blob_exists(blob_name):
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tmp_out_path = blob_storage.download_file(blob_name, tmpfile)
            with open(tmp_out_path, "r") as file:
                json_data = json.load(file)
                config_videos = json_data.get("videos", [])
        else:
            json_data = {}
            config_videos = []

        for audio in audios:
            filename = audio["filename"]
            audio_b64_str = audio["audio"]
            transcript_b64_str = audio["transcript"]
            audio_path = base64_to_path(audio_b64_str, ".wav")
            transcript_path = base64_to_path(transcript_b64_str, ".txt")
            audio_name = f"{avatar_name}_{os.path.splitext(filename)[0]}"

            with open(transcript_path, "r") as file:
                speech_start = get_speech_start_sec(audio_path)
                audio_duration = get_audio_len_sec(audio_path)

                transcript_text = file.read()

                for cfg_video in config_videos:
                    if cfg_video["name"] == f"{audio_name}.mp4":
                        audio_name = f"{audio_name}-{gen_random_suffix()}"
                        log.warning(
                            f"There is already a video named {cfg_video['name']}, changing name to {audio_name}"
                        )
                        break

                config_videos = [
                    cfg_video
                    for cfg_video in config_videos
                    if cfg_video["pregenerated_text"] != transcript_text
                ]

                config_dict = {
                    "name": f"{audio_name}.mp4",
                    "speech_start": speech_start,
                    "speech_duration": round(audio_duration - speech_start, 2),
                    "video_duration": audio_duration,
                    "pregenerated_text": transcript_text,
                    "audio_file": f"{audio_name}.wav",
                }

                log.info(f"Config dict: {config_dict}")
                config_videos.append(config_dict)

            video_path = f"{result_dir}/{audio_name}.mp4"

            log.info(f"Generating video for audio {filename}")
            cmd = f'python inference.py --driven_audio {audio_path} --source_image {source_image} --result_dir {result_dir}{f" --bg_image {bg_image}" if bg_image else ""}{f" --ref_eyeblink {ref_blink}" if ref_blink else ""}{f" --ref_pose {ref_pose}" if ref_pose else ""} --final_vid_name {audio_name}.mp4{" --still" if is_still_mode else ""} --preprocess {preprocess_type} --expression_scale {exp_scale} --head_motion_scale {head_motion_scale} --batch_size {batch_size} --size 512 --enhancer gfpgan'
            os.system(cmd)
            log.info(f"Video generated and saved to {video_path}")

            blob_storage.upload_file(video_path, avatar_name)
            log.info(f"Video {video_path} uploaded to storage")

            blob_storage.upload_file(
                audio_path, avatar_name, custom_name=f"{audio_name}.wav"
            )
            log.info(f"Audio {filename} uploaded to storage")

        json_data["videos"] = config_videos
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_json_file:
            tmp_json_file.write(json.dumps(json_data).encode())

        blob_storage.upload_file(
            tmp_json_file.name, avatar_name, custom_name=config_filename
        )
        log.info(f"Video config {config_filename} uploaded to storage")

        return None
