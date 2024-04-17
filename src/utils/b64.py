import base64
from PIL import Image
import io
import tempfile


def base64_to_image(input_str):
    try:
        if ';base64,' in input_str:
            format, base64_str = input_str.split(';base64,')
        else:
            format = 'image/png'
            base64_str = input_str

        binary_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(binary_data))
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        format_type = format.split('/')[-1]

        tmpfile.name += f'.{format_type}'

        image.save(tmpfile.name, format_type)
        return tmpfile.name
    except Exception as e:
        print(f"The input is neither a valid file path nor a base64 string. Error: {e}")
        return None


def base64_to_audio(input_str):
    try:
        if ';base64,' in input_str:
            _, base64_str = input_str.split(';base64,')
        else:
            base64_str = input_str

        binary_data = base64.b64decode(base64_str)

        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmpfile.write(binary_data)
        return tmpfile.name
    except Exception as e:
        print(f"The input is neither a valid file path nor a base64 string. Error: {e}")
        return None


def base64_to_video(input_str):
    try:
        if ';base64,' in input_str:
            _, base64_str = input_str.split(';base64,')
        else:
            base64_str = input_str

        binary_data = base64.b64decode(base64_str)

        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmpfile.write(binary_data)
        return tmpfile.name
    except Exception as e:
        print(f"The input is neither a valid file path nor a base64 string. Error: {e}")
        return None


def path_to_base64(file_path):
    try:
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        print(f"The input is not a valid file path, received {file_path}")
        return None
