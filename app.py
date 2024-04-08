import os, sys
import gradio as gr
from src.gradio_demo import SadTalker

try:
    import webui  # in webui

    in_webui = True
except:
    in_webui = False


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


def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config'):
    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False, title="Avatar Generator 😎", theme="gradio/monochrome") as sadtalker_interface:
        with gr.Row():
            gr.Markdown("<div style='display: flex;justify-content: center'> <h1> Generate Talking Avatar 🗣️</h1> </div>")

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

                with gr.Tabs(elem_id="sadtalker_generated"):
                    with gr.TabItem('Test video'):
                        gen_video = gr.Video(label="Generated video", format="mp4", width=256)

                with gr.Tabs(elem_id="generate_avatar"):
                    with gr.TabItem('Generate avatar'):
                        gr.Markdown(
                            "<div <p style='color: #A3A3A3;'>When you are happy with the test video, generate complete avatar</p> </div>")
                        generate = gr.Button('Generate Avatar', elem_id="sadtalker_generate", variant='primary')

        inputs = [
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
        ]

        test.click(
            fn=sad_talker.test,
            inputs=inputs,
            outputs=[gen_video]
        )

        generate.click(
            fn=sad_talker.generate_avatar,
            inputs=inputs,
        )

    return sadtalker_interface


if __name__ == "__main__":
    demo = sadtalker_demo()
    demo.queue()
    demo.launch(inbrowser=True, favicon_path='./favicon.png')
