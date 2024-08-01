import os
import asyncio
import re
import logging as log
import time

from fastapi import FastAPI
import uvicorn
import gradio as gr
from dotenv import load_dotenv
from rq import Queue
from redis import Redis
from rq_dashboard_fast import RedisQueueDashboard

from src.api.models import GenerateRequest
from src.gradio_demo import SadTalker
from src.utils.b64 import path_to_base64, base64_to_path

try:
    import webui  # in webui

    in_webui = True
except:
    in_webui = False

log_format = "%(asctime)s [%(levelname)s] - %(message)s"
log.basicConfig(level=log.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")


def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)


def is_valid_email(email):
    email_regex = r"^([a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3})$"
    if re.search(email_regex, email):
        return True
    return False


def is_valid_av_name(name):
    name_regex = r"^[A-Za-z0-9\-_]+$"
    if re.search(name_regex, name):
        return True
    return False


def gen_avatar_job(
    source_image,
    bg_image,
    ref_blink,
    ref_pose,
    preprocess_type,
    is_still_mode,
    exp_scale,
    head_motion_scale,
    email,
    avatar_name,
):

    if not avatar_name or not is_valid_av_name(avatar_name):
        error_message = (
            "Please enter a valid avatar name (letters, digits, hyphen, underscore)."
        )
        return gr.Markdown(
            f"<div><p style='color: red;'>{error_message}</p></div>", visible=True
        ), gr.Button(interactive=True)

    if not email or not is_valid_email(email):
        error_message = "Please enter a valid email address."
        return gr.Markdown(
            f"<div><p style='color: red;'>{error_message}</p></div>", visible=True
        ), gr.Button(interactive=True)

    source_img_b64 = path_to_base64(source_image)
    bg_img_b64 = path_to_base64(bg_image)
    ref_blink_b64 = path_to_base64(ref_blink)
    ref_pose_b64 = path_to_base64(ref_pose)

    job = q.enqueue(
        sad_talker.generate_avatar,
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
        result_ttl=-1,
        job_timeout="24h",
    )
    return gr.Markdown(
        f"<div> <p style='color: #A3A3A3; margin: 5px 0'> Avatar generation started! 🥳 </p> <p style='color: #A3A3A3; margin: 5px 0'> We will send you a link to your email when the avatar is generated. </p> <p style='color: #A3A3A3; margin: 5px 0'> You can track the progress <a href='{API_BASE_URI}/rq/job/{job.get_id()}' target='_blank'> <b>here</b> </a> </p> </div>",
        visible=True,
    ), gr.Button(interactive=False)


def gen_custom_videos_job(
    source_image,
    bg_image,
    ref_blink,
    ref_pose,
    preprocess_type,
    is_still_mode,
    exp_scale,
    head_motion_scale,
    avatar_name,
    uploaded_files,
):

    if not avatar_name or not is_valid_av_name(avatar_name):
        error_message = (
            "Please enter a valid avatar name (letters, digits, hyphen, underscore)."
        )
        return gr.Markdown(
            f"<div><p style='color: red;'>{error_message}</p></div>", visible=True
        ), gr.Button(interactive=True)

    source_img_b64 = path_to_base64(source_image)
    bg_img_b64 = path_to_base64(bg_image)
    ref_blink_b64 = path_to_base64(ref_blink)
    ref_pose_b64 = path_to_base64(ref_pose)

    audios = []

    uploaded_audios = list(filter(lambda file: file.endswith(".wav"), uploaded_files))
    uploaded_txt_files = list(
        filter(lambda file: file.endswith(".txt"), uploaded_files)
    )

    for audio_path in uploaded_audios:
        filename = os.path.basename(audio_path)
        txt_filename = filename.replace(".wav", ".txt")

        try:
            txt_filepath = [
                path
                for path in uploaded_txt_files
                if os.path.basename(path) == txt_filename
            ][0]
        except IndexError:
            error_message = (
                f"Missing a transcript file {txt_filename}. Please upload it."
            )
            return gr.Markdown(
                f"<div><p style='color: red;'>{error_message}</p></div>", visible=True
            ), gr.Button(interactive=True)

        audio_dict = {
            "filename": filename,
            "audio": path_to_base64(audio_path),
            "transcript": path_to_base64(txt_filepath),
        }
        audios.append(audio_dict)

    job = q.enqueue(
        sad_talker.gen_custom_videos,
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
        result_ttl=86400,
        job_timeout="24h",
    )

    return gr.Markdown(
        f"<div> <p style='color: #A3A3A3; margin: 5px 0'> Custom videos generation started! </p> <p style='color: #A3A3A3; margin: 5px 0'> You can track the progress <a href='{API_BASE_URI}/rq/job/{job.get_id()}' target='_blank'> <b>here</b> </a> </p> </div>",
        visible=True,
    ), gr.Button(interactive=False)


def gen_test_video(
    source_image,
    driven_audio,
    preprocess_type,
    is_still_mode,
    enhancer,
    batch_size,
    size_of_image,
    pose_style,
    exp_scale,
    head_motion_scale,
    bg_image,
    ref_blink,
    ref_pose,
):
    source_img_b64 = path_to_base64(source_image)
    bg_img_b64 = path_to_base64(bg_image)
    driven_audio_b64 = path_to_base64(driven_audio)
    ref_blink_b64 = path_to_base64(ref_blink)
    ref_pose_b64 = path_to_base64(ref_pose)

    job = q.enqueue(
        sad_talker.test,
        source_img_b64,
        driven_audio_b64,
        preprocess_type,
        is_still_mode,
        enhancer,
        batch_size,
        size_of_image,
        pose_style,
        exp_scale,
        head_motion_scale,
        bg_img_b64,
        ref_blink_b64,
        ref_pose_b64,
        result_ttl=86400,
        job_timeout="5h",
    )

    while True:
        time.sleep(5)
        result = job.latest_result()
        if result:
            if result.return_value:
                vid_path = base64_to_path(result.return_value, ".mp4")
                log.info(f"Video saved to: {vid_path}")
                return vid_path
            else:
                log.error(f"Exception: {result.exc_string}")
                raise Exception(f"Job failed with error: {result.exc_string}")


def sadtalker_demo():
    with gr.Blocks(
        analytics_enabled=False, title="Avatar Generator 😎", theme="gradio/monochrome"
    ) as sadtalker_interface:
        with gr.Row():
            gr.Markdown(
                "<div style='display: flex;justify-content: center'> <h1> Generate Talking Avatar 🗣️</h1> </div>"
            )

        with gr.Row():
            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem("Source image"):
                        with gr.Row():
                            source_image = gr.Image(
                                label="Upload image in 1:1 ratio",
                                sources=["upload"],
                                type="filepath",
                                elem_id="img2img_image",
                                width=512,
                            )
                with gr.Tabs(elem_id="sadtalker_bg_image"):
                    with gr.TabItem("Background image"):
                        with gr.Row():
                            bg_image = gr.Image(
                                label="Upload image in 16:9 or 3:4 ratio",
                                sources=["upload"],
                                type="filepath",
                                elem_id="img2img_bg_image",
                                width=512,
                            )
                with gr.Tabs(elem_id="sadtalker_ref_video"):
                    with gr.TabItem("Eyeblink reference video"):
                        with gr.Row():
                            ref_blink = gr.Video(
                                label="Upload eyeblink video",
                                format="mp4",
                                sources=["upload"],
                                elem_id="img2img_ref_blink",
                                width=512,
                            )
                    with gr.TabItem("Pose reference video"):
                        with gr.Row():
                            ref_pose = gr.Video(
                                label="Upload pose video",
                                format="mp4",
                                sources=["upload"],
                                elem_id="img2img_ref_pose",
                                width=512,
                            )

                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem("Input audio"):
                        with gr.Column(variant="panel"):
                            driven_audio = gr.Audio(
                                label="Upload wav audio or use microphone",
                                sources=["upload", "microphone"],
                                type="filepath",
                            )

            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem("Settings"):
                        with gr.Column(variant="panel"):
                            pose_style = gr.Slider(
                                minimum=0,
                                maximum=46,
                                step=1,
                                label="Pose style",
                                value=0,
                                visible=False,
                            )  #
                            size_of_image = gr.Radio(
                                [256, 512],
                                value=256,
                                label="Face model resolution",
                                info="Use 256/512 model?",
                            )  #
                            preprocess_type = gr.Radio(
                                ["resize", "full", "crop", "extcrop", "extfull"],
                                value="resize",
                                label="Preprocess",
                                info="How to handle input image?",
                            )
                            is_still_mode = gr.Checkbox(
                                label="Still Mode (fewer head motion, works with preprocess `full`)",
                                value=True,
                            )
                            batch_size = gr.Slider(
                                label="batch size in generation",
                                step=1,
                                maximum=10,
                                value=2,
                                visible=False,
                            )
                            enhancer = gr.Checkbox(
                                label="GFPGAN as Face enhancer",
                                value=True,
                                visible=False,
                            )
                            exp_scale = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                step=0.1,
                                label="Expression scale",
                                value=1.0,
                            )
                            head_motion_scale = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                step=0.1,
                                label="Head motion scale",
                                value=1.0,
                            )
                            test = gr.Button(
                                "Test", elem_id="sadtalker_test", variant="secondary"
                            )

                with gr.Tabs(elem_id="sadtalker_test_video"):
                    with gr.TabItem("Test video"):
                        gen_video = gr.Video(
                            label="Generated video", format="mp4", width=256
                        )

                with gr.Tabs(elem_id="gen_video_tabs"):
                    with gr.TabItem("Generate avatar"):
                        gr.Markdown(
                            "<div <p style='color: #A3A3A3;'>When you are happy with the test video, generate a complete avatar. The result will be sent to your email.</p> </div>"
                        )
                        avatar_name = gr.Textbox(
                            label="Avatar name",
                            type="text",
                            placeholder="Choose a name for your avatar",
                            max_lines=1,
                        )
                        email = gr.Textbox(
                            label="Email",
                            type="email",
                            placeholder="Input your email",
                            max_lines=1,
                        )
                        generate_btn = gr.Button(
                            "Generate Avatar", variant="primary", interactive=True
                        )
                        job_result = gr.Markdown(visible=False)

                    with gr.TabItem("Generate custom videos"):
                        with gr.Column(variant="panel"):
                            gr.Markdown(
                                "<div <p style='color: #A3A3A3;'>Upload wav recordings and txt files with transcript to create custom avatar videos.</p> </div>"
                            )
                            uploaded_files = gr.Files(
                                label="Upload custom wav files for existing avatar",
                                file_types=["wav", "txt"],
                                type="filepath",
                            )
                            av_name = gr.Textbox(
                                label="Avatar name",
                                type="text",
                                placeholder="Choose an existing avatar name",
                                max_lines=1,
                            )
                            gen_custom_vid_btn = gr.Button(
                                "Generate custom videos",
                                variant="primary",
                                interactive=True,
                            )
                            custom_videos_job_result = gr.Markdown(visible=False)

        test.click(
            fn=gen_test_video,
            inputs=[
                source_image,
                driven_audio,
                preprocess_type,
                is_still_mode,
                enhancer,
                batch_size,
                size_of_image,
                pose_style,
                exp_scale,
                head_motion_scale,
                bg_image,
                ref_blink,
                ref_pose,
            ],
            outputs=[gen_video],
        )

        generate_btn.click(
            fn=gen_avatar_job,
            inputs=[
                source_image,
                bg_image,
                ref_blink,
                ref_pose,
                preprocess_type,
                is_still_mode,
                exp_scale,
                head_motion_scale,
                email,
                avatar_name,
            ],
            outputs=[job_result, generate_btn],
        )

        gen_custom_vid_btn.click(
            fn=gen_custom_videos_job,
            inputs=[
                source_image,
                bg_image,
                ref_blink,
                ref_pose,
                preprocess_type,
                is_still_mode,
                exp_scale,
                head_motion_scale,
                av_name,
                uploaded_files,
            ],
            outputs=[custom_videos_job_result, gen_custom_vid_btn],
        )

    return sadtalker_interface


if __name__ == "__main__":
    load_dotenv()

    GRADIO_HOST = os.environ.get("GRADIO_HOST", "0.0.0.0")
    GRADIO_PORT = int(os.environ.get("GRADIO_PORT", "7860"))

    API_BASE_URI = os.environ.get("API_BASE_URI", "http://localhost:9988")
    API_HOST = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT = int(os.environ.get("API_PORT", "9988"))

    REDIS_HOST = os.environ.get("REDIS_HOST", "sadtalker-redis")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

    q = Queue(connection=Redis(host=REDIS_HOST, port=REDIS_PORT))

    log.info(f"Application running on port {GRADIO_PORT}")

    sad_talker = SadTalker(checkpoint_path="checkpoints", config_path="src/config")
    demo = sadtalker_demo()
    demo.queue()

    app = FastAPI()

    dashboard = RedisQueueDashboard(f"redis://{REDIS_HOST}:{REDIS_PORT}", "/rq")
    app.mount("/rq", dashboard)

    @app.get("/health")
    def health():
        return {"message": "App is running"}

    @app.get("/status/{job_id}")
    def status(job_id: str):
        job = q.fetch_job(job_id)
        if job is None:
            return {"status": "unknown"}
        if job.is_failed:
            return {"status": "failed"}
        return {"status": job.get_status(), "result": job.return_value()}

    @app.post("/generate")
    def generate(r: GenerateRequest):
        source_img_b64 = r.source_image
        bg_img_b64 = r.bg_image if r.bg_image else None
        ref_blink_b64 = r.ref_blink if r.ref_blink else None
        ref_pose_b64 = r.ref_pose if r.ref_pose else None
        preprocess_type = r.preprocess_type
        is_still_mode = r.is_still_mode
        exp_scale = r.exp_scale
        head_motion_scale = r.head_motion_scale
        email = r.email
        avatar_name = r.avatar_name

        job = q.enqueue(
            sad_talker.generate_avatar,
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
            result_ttl=-1,
            job_timeout="24h",
        )

        return {"job_id": job.get_id()}

    loop = asyncio.get_event_loop()

    def run_gradio():
        demo.launch(
            inbrowser=True,
            favicon_path="./favicon.png",
            server_name=GRADIO_HOST,
            server_port=GRADIO_PORT,
        )

    loop.run_in_executor(None, run_gradio)

    uvicorn.run(app, host=API_HOST, port=API_PORT)
