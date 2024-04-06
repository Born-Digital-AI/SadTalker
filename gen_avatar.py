import os

audio_dir = './examples/custom/audio'
src_img_path = './examples/custom/fatima.png'
src_bg_img_path = './examples/custom/fatima_wide.png'
result_dir = './examples/custom/fatima'

os.makedirs(result_dir, exist_ok=True)

for audio in os.listdir(audio_dir):
    audio_path = os.path.join(audio_dir, audio)
    audio_name = os.path.splitext(audio)[0]

    print(f'Generating video for audio {audio}')
    cmd = f'python inference.py --driven_audio {audio_path} --source_image {src_img_path} --result_dir {result_dir} --bg_image {src_bg_img_path} --final_vid_name {audio_name}.mp4 --still --preprocess resize --expression_scale 0.8 --batch_size 2 --size 512 --enhancer gfpgan'
    os.system(cmd)
    print(f'Video {audio_name}.mp4 generated for audio {audio}')

print(f'Generating default video')
default_vid_name = 'default-video.mp4'
cmd = f'python inference.py --source_image {src_img_path} --result_dir {result_dir} --bg_image {src_bg_img_path} --final_vid_name {default_vid_name} --still --preprocess resize --expression_scale 0.8 --batch_size 2 --size 512 --enhancer gfpgan --idlemode --len 30'
os.system(cmd)
print(f'Video {default_vid_name} generated')
