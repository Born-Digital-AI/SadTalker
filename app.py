import os
import asyncio
import re
import logging as log

from fastapi import FastAPI
import uvicorn
import gradio as gr
from dotenv import load_dotenv
from rq import Queue
from redis import Redis
from rq_dashboard_fast import RedisQueueDashboard

from src.api.models import GenerateRequest
from src.gradio_demo import SadTalker
from src.utils.b64 import image_to_base64

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
    email_regex = "^([a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3})$"
    if re.search(email_regex, email):
        return True
    return False


def gen_avatar_job(source_image,
                   bg_image,
                   preprocess_type,
                   is_still_mode,
                   exp_scale,
                   email):
    if email and is_valid_email(email):
        source_img_b64 = image_to_base64(source_image)
        bg_img_b64 = image_to_base64(bg_image)

        job = q.enqueue(sad_talker.generate_avatar, source_img_b64, bg_img_b64, preprocess_type, is_still_mode,
                        exp_scale, email, result_ttl=86400)
        return gr.Markdown(
            f"<div> <p style='color: #A3A3A3; margin: 5px 0'> Avatar generation started! 🥳 </p> <p style='color: #A3A3A3; margin: 5px 0'> We will send you a link to your email when the avatar is generated. </p> <p style='color: #A3A3A3; margin: 5px 0'> You can track the progress <a href='{API_BASE_URI}/rq/job/{job.get_id()}' target='_blank'> <b>here</b> </a> </p> </div>",
            visible=True), gr.Button(interactive=False)
    else:
        error_message = "Please enter a valid email address."
        return gr.Markdown(f"<div><p style='color: red;'>{error_message}</p></div>", visible=True), gr.Button(interactive=True)


def sadtalker_demo(sad_talker):
    with (gr.Blocks(analytics_enabled=False, title="Avatar Generator 😎",
                    theme="gradio/monochrome") as sadtalker_interface):
        with gr.Row():
            gr.Markdown(
                "<div style='display: flex;justify-content: center'> <h1> Generate Talking Avatar 🗣️</h1> </div>")

        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            source_image = gr.Image(label="Source image", sources=["upload"], type="filepath",
                                                    elem_id="img2img_image", width=512)
                with gr.Tabs(elem_id="sadtalker_bg_image"):
                    with gr.TabItem('Upload background image'):
                        with gr.Row():
                            bg_image = gr.Image(label="Background image", sources=["upload"], type="filepath",
                                                elem_id="img2img_bg_image", width=512)

                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Upload OR TTS'):
                        with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="Input audio", sources=["upload", "microphone"],
                                                    type="filepath")

            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('Settings'):
                        with gr.Column(variant='panel'):
                            pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0,
                                                   visible=False)  #
                            size_of_image = gr.Radio([256, 512], value=256, label='face model resolution',
                                                     info="use 256/512 model?")  #
                            preprocess_type = gr.Radio(['resize', 'full', 'crop', 'extcrop', 'extfull'], value='resize',
                                                       label='preprocess', info="How to handle input image?")
                            is_still_mode = gr.Checkbox(
                                label="Still Mode (fewer head motion, works with preprocess `full`)", value=True)
                            batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=2,
                                                   visible=False)
                            enhancer = gr.Checkbox(label="GFPGAN as Face enhancer", value=True, visible=False)
                            exp_scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Expression scale",
                                                  value=1.0)
                            test = gr.Button('Test', elem_id="sadtalker_test", variant='secondary')

                with gr.Tabs(elem_id="sadtalker_test_video"):
                    with gr.TabItem('Test video'):
                        gen_video = gr.Video(label="Generated video", format="mp4", width=256)

                with gr.Tabs(elem_id="generate_avatar"):
                    with gr.TabItem('Generate avatar'):
                        gr.Markdown(
                            "<div <p style='color: #A3A3A3;'>When you are happy with the test video, generate a complete avatar. The result will be sent to your email.</p> </div>")
                        email = gr.Textbox(label="Email", type="email", placeholder="Input your email", max_lines=1)
                        generate_btn = gr.Button('Generate Avatar', elem_id="generate_btn", variant='primary',
                                                 interactive=True)
                        job_result = gr.Markdown(visible=False)

        test.click(
            fn=sad_talker.test,
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
                bg_image
            ],
            outputs=[gen_video]
        )

        generate_btn.click(
            fn=gen_avatar_job,
            inputs=[
                source_image,
                bg_image,
                preprocess_type,
                is_still_mode,
                exp_scale,
                email
            ],
            outputs=[job_result, generate_btn]
        )

    return sadtalker_interface


if __name__ == "__main__":
    load_dotenv()

    GRADIO_HOST = os.environ.get('GRADIO_HOST', '0.0.0.0')
    GRADIO_PORT = int(os.environ.get('GRADIO_PORT', '7860'))

    API_BASE_URI = os.environ.get('API_BASE_URI', 'http://localhost:9988')
    API_HOST = os.environ.get('API_HOST', '0.0.0.0')
    API_PORT = int(os.environ.get('API_PORT', '9988'))

    REDIS_HOST = os.environ.get('REDIS_HOST', 'sadtalker-redis')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))

    q = Queue(connection=Redis(host=REDIS_HOST, port=REDIS_PORT))

    log.info(f"Application running on port {GRADIO_PORT}")

    sad_talker = SadTalker(checkpoint_path='checkpoints', config_path='src/config', lazy_load=True)
    demo = sadtalker_demo(sad_talker)
    demo.queue()

    app = FastAPI()

    dashboard = RedisQueueDashboard(f"redis://{REDIS_HOST}:{REDIS_PORT}", "/rq")
    app.mount("/rq", dashboard)


    @app.get('/health')
    def health():
        return {"message": "App is running"}


    @app.get('/status/{job_id}')
    def status(job_id: str):
        job = q.fetch_job(job_id)
        if job is None:
            return {"status": "unknown"}
        if job.is_failed:
            return {"status": "failed"}
        return {"status": job.get_status(), "result": job.return_value()}


    @app.post('/generate')
    def generate(r: GenerateRequest):
        source_img_b64 = r.source_image
        bg_img_b64 = r.bg_image if r.bg_image else None
        preprocess_type = r.preprocess_type
        is_still_mode = r.is_still_mode
        exp_scale = r.exp_scale
        email = r.email

        job = q.enqueue(sad_talker.generate_avatar, source_img_b64, bg_img_b64, preprocess_type, is_still_mode,
                        exp_scale, email, result_ttl=86400)

        return {"job_id": job.get_id()}


    loop = asyncio.get_event_loop()


    def run_gradio():
        demo.launch(inbrowser=True, favicon_path='./favicon.png', server_name=GRADIO_HOST, server_port=GRADIO_PORT)


    loop.run_in_executor(None, run_gradio)

    uvicorn.run(app, host=API_HOST, port=API_PORT)
