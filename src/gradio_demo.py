import torch, uuid
import os, shutil
from pydub import AudioSegment
from rq import get_current_job
import logging as log

from src.utils.b64 import base64_to_image
from src.utils.email_sender import EmailSender
from src.utils.blob_storage import BlobStorage
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path


def mp3_to_wav(mp3_filename, wav_filename, frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename, format="wav")


class SadTalker():

    def __init__(self, checkpoint_path='checkpoints', config_path='src/config', lazy_load=False):

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.device = device

        os.environ['TORCH_HOME'] = checkpoint_path

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path

    def test(self, source_image, driven_audio, preprocess='crop',
             still_mode=False, use_enhancer=False, batch_size=1, size=256,
             pose_style=0, exp_scale=1.0, bg_image=None,
             use_ref_video=False,
             ref_video=None,
             ref_info=None,
             use_idle_mode=False,
             length_of_audio=0, use_blink=True,
             result_dir='./results/'):

        self.sadtalker_paths = init_path(self.checkpoint_path, self.config_path, size, False, preprocess)
        log.info(self.sadtalker_paths)

        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)

        time_tag = str(uuid.uuid4())
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)

        input_dir = os.path.join(save_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        log.info(source_image)
        pic_path = os.path.join(input_dir, os.path.basename(source_image))
        shutil.move(source_image, input_dir)

        if driven_audio is not None and os.path.isfile(driven_audio):
            audio_path = os.path.join(input_dir, os.path.basename(driven_audio))

            #### mp3 to wav
            if '.mp3' in audio_path:
                mp3_to_wav(driven_audio, audio_path.replace('.mp3', '.wav'), 16000)
                audio_path = audio_path.replace('.mp3', '.wav')
            else:
                shutil.copy(driven_audio, input_dir)

        elif use_idle_mode:
            audio_path = os.path.join(input_dir, 'idlemode_' + str(
                length_of_audio) + '.wav')  ## generate audio from this new audio_path
            from pydub import AudioSegment
            one_sec_segment = AudioSegment.silent(duration=1000 * length_of_audio)  # duration in milliseconds
            one_sec_segment.export(audio_path, format="wav")
        else:
            assert use_ref_video == True and ref_info == 'all'

        if use_ref_video and ref_info == 'all':  # full ref mode
            ref_video_videoname = os.path.basename(ref_video)
            audio_path = os.path.join(save_dir, ref_video_videoname + '.wav')
            log.info(f'new audiopath: {audio_path}')
            # if ref_video contains audio, set the audio from ref_video.
            cmd = r"ffmpeg -y -hide_banner -loglevel error -i %s %s" % (ref_video, audio_path)
            os.system(cmd)

        os.makedirs(save_dir, exist_ok=True)

        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(pic_path, first_frame_dir,
                                                                                    preprocess, True, size)

        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        if use_ref_video:
            log.info('using ref video for genreation')
            ref_video_videoname = os.path.splitext(os.path.split(ref_video)[-1])[0]
            ref_video_frame_dir = os.path.join(save_dir, ref_video_videoname)
            os.makedirs(ref_video_frame_dir, exist_ok=True)
            log.info('3DMM Extraction for the reference video providing pose')
            ref_video_coeff_path, _, _ = self.preprocess_model.generate(ref_video, ref_video_frame_dir, preprocess,
                                                                        source_image_flag=False)
        else:
            ref_video_coeff_path = None

        if use_ref_video:
            if ref_info == 'pose':
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = None
            elif ref_info == 'blink':
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'pose+blink':
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'all':
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = None
            else:
                raise ('error in refinfo')
        else:
            ref_pose_coeff_path = None
            ref_eyeblink_coeff_path = None

        # audio2ceoff
        if use_ref_video and ref_info == 'all':
            coeff_path = ref_video_coeff_path  # self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
        else:
            batch = get_data(first_coeff_path, audio_path, self.device, ref_eyeblink_coeff_path=ref_eyeblink_coeff_path,
                             still=still_mode, idlemode=use_idle_mode, length_of_audio=length_of_audio,
                             use_blink=use_blink)  # longer audio?
            coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

        # coeff2video
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size,
                                   still_mode=still_mode, preprocess=preprocess, size=size, expression_scale=exp_scale)
        return_path = self.animate_from_coeff.generate(data, save_dir, pic_path, crop_info,
                                                       enhancer='gfpgan' if use_enhancer else None,
                                                       preprocess=preprocess, img_size=size, bg_image=bg_image)
        video_name = data['video_name']
        log.info(f'The generated video is named {video_name} in {save_dir}')

        del self.preprocess_model
        del self.audio_to_coeff
        del self.animate_from_coeff

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import gc;
        gc.collect()

        return return_path

    def generate_avatar(self, source_img_b64,
                        bg_img_b64,
                        preprocess_type,
                        is_still_mode,
                        exp_scale,
                        email):
        job = get_current_job()
        job_id = job.id

        job.meta['job_id'] = job_id
        job.meta['email'] = email
        job.meta['preprocess_type'] = preprocess_type
        job.meta['is_still_mode'] = is_still_mode
        job.meta['exp_scale'] = exp_scale
        job.save_meta()

        source_image = base64_to_image(source_img_b64)
        bg_image = base64_to_image(bg_img_b64)

        log.info(f"Source image path: {source_image}")
        log.info(f"Background image path: {bg_image}")

        audio_dir = './examples/custom/audio'
        result_dir = f'./examples/custom/result/{job_id}'
        output_dir = './examples/custom/out'
        batch_size = os.environ.get('BATCH_SIZE', 2)

        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        for audio in os.listdir(audio_dir):
            audio_path = os.path.join(audio_dir, audio)
            audio_name = os.path.splitext(audio)[0]

            log.info(f'Generating video for audio {audio}')
            cmd = f'python inference.py --driven_audio {audio_path} --source_image {source_image} --result_dir {result_dir}{f" --bg_image {bg_image}" if bg_image else ""} --final_vid_name {audio_name}.mp4{" --still" if is_still_mode else ""} --preprocess {preprocess_type} --expression_scale {exp_scale} --batch_size {batch_size} --size 512 --enhancer gfpgan'
            os.system(cmd)
            log.info(f'Video generated and saved to {result_dir}/{audio_name}.mp4')

        log.info(f'Generating default video')
        default_vid_name = 'default-video.mp4'
        cmd = f'python inference.py --source_image {source_image} --result_dir {result_dir}{f" --bg_image {bg_image}" if bg_image else ""} --final_vid_name {default_vid_name}{" --still" if is_still_mode else ""} --preprocess {preprocess_type} --expression_scale {exp_scale} --batch_size {batch_size} --size 512 --enhancer gfpgan --idlemode --len 30'
        os.system(cmd)
        log.info(f'Default video generated and saved to {result_dir}/{default_vid_name}')

        output_filename = f'{output_dir}/{job_id}'
        shutil.make_archive(output_filename, 'zip', result_dir)
        log.info(f'Archive created at {output_filename}.zip')

        blob_storage = BlobStorage()
        blob_storage.upload_file(f'{output_filename}.zip', job_id)
        download_url = blob_storage.get_file_url()

        email_sender = EmailSender()
        subject = 'Link to your avatar videos'
        html_body = f"""
                <html>
                <body>
                    <h1>Hello there,</h1>
                    <p>Your avatar videos are ready! 🎉</p>
                    <p>You can download them by clicking the link below:</p>
                    <a href="{download_url}" target="_blank">Download Videos</a>
                    <p>Enjoy your new avatar! 🤖</p>
                </body>
                </html>
                """
        email_sender.send(subject, html_body, [email])

        return download_url
