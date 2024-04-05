import shutil
import uuid

import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm


def fill_images(imgs, base_img_path):
    overlay_width, overlay_height = Image.fromarray(imgs[0]).size

    base_img = Image.open(base_img_path).convert("RGBA")
    enhancer = ImageEnhance.Brightness(base_img)
    base_img = enhancer.enhance(0.97)
    base_width, base_height = base_img.size
    is_wide_format = base_width > base_height

    base_orig_width = base_width * 256 // base_height if is_wide_format else 256
    base_orig_height = 256 if is_wide_format else base_height * 256 // base_width

    base_img = base_img.resize((base_orig_width, base_orig_height)) # lower quality of base image to match overlay img

    base_width = base_width * overlay_height // base_height
    base_height = overlay_height

    base_img = base_img.resize((base_width, base_height))

    overlay_images = []

    for index, frame in enumerate(tqdm(imgs, "Overlay images gen:")):
        overlay_img = Image.fromarray(frame).convert("RGBA")
        position = ((base_width - overlay_width) // 2, (base_height - overlay_height) // 2)

        # portrait video
        if not is_wide_format:
            border_top = 20
            border_bottom = 120
            mask = np.zeros((overlay_height, overlay_width))
            for y in range(overlay_height):
                for x in range(overlay_width):
                    dist_y_top = y
                    dist_y_bottom = overlay_height - y - 1

                    factor_top = 1 if index < 10 else (dist_y_top / border_top)
                    factor_bottom = 1 if index < 10 else (dist_y_bottom / border_bottom)
                    if dist_y_top > border_top and dist_y_bottom > border_bottom:
                        mask[y][x] = 255
                    elif dist_y_top <= border_top:
                        mask[y][x] = 255 * factor_top
                    else:
                        mask[y][x] = 255 * factor_bottom
        # wide video
        else:
            border = 200
            mask = np.zeros((overlay_height, overlay_width))
            for y in range(overlay_height):
                for x in range(overlay_width):
                    dist_x = min(x, overlay_width - x)

                    factor = dist_x / border
                    if dist_x > border:
                        mask[y][x] = 255
                    else:
                        mask[y][x] = 255 * factor

        mask = Image.fromarray(np.uint8(mask), 'L')
        base_img.paste(overlay_img, position, mask)

        open_cv_image = np.array(base_img)
        overlay_images.append(open_cv_image)

    return overlay_images


def load_video_to_cv2(input_path):
    video_stream = cv2.VideoCapture(input_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        full_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return full_frames


def save_video_with_watermark(video, audio, save_path, watermark=False):
    temp_file = str(uuid.uuid4()) + '.mp4'
    cmd = r'ffmpeg -y -hide_banner -loglevel error -i "%s" -i "%s" -vcodec copy "%s"' % (video, audio, temp_file)
    os.system(cmd)

    if watermark is False:
        shutil.move(temp_file, save_path)
    else:
        # watermark
        try:
            ##### check if stable-diffusion-webui
            import webui
            from modules import paths
            watarmark_path = paths.script_path + "/extensions/SadTalker/docs/sadtalker_logo.png"
        except:
            # get the root path of sadtalker.
            dir_path = os.path.dirname(os.path.realpath(__file__))
            watarmark_path = dir_path + "/../../docs/sadtalker_logo.png"

        cmd = r'ffmpeg -y -hide_banner -loglevel error -i "%s" -i "%s" -filter_complex "[1]scale=100:-1[wm];[0][wm]overlay=(main_w-overlay_w)-10:10" "%s"' % (
        temp_file, watarmark_path, save_path)
        os.system(cmd)
        os.remove(temp_file)
