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


def image_to_base64(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        print(f"The input is not a valid image_path, received {image_path}")
        return None
