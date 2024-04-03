from PIL import Image
import time
import cv2
from skimage.util import img_as_ubyte

from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
import torch
from basicsr.utils import imwrite, img2tensor, tensor2img
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import numpy as np

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}


def codeformer_enhance(predictions):
    device = get_device()
    w = 0.5
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                          connect_list=['32', '64', '128', '256']).to(device)

    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
                                   model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    face_helper = FaceRestoreHelper(
        upscale_factor=2,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device)

    image_ts = []
    for idx in range(predictions.shape[0]):
        prediction = predictions[idx]
        image = np.transpose(prediction.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
        image = img_as_ubyte(image)

        face_helper.read_image(image)
        face_helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()

        image_t = img2tensor(image / 255., bgr2rgb=False, float32=True)
        normalize(image_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        image_t = image_t.unsqueeze(0)
        image_t = F.interpolate(image_t, size=(512, 512))

        image_ts.append(image_t)

    input_batch = torch.cat(image_ts)
    input_batch = input_batch.to(device)

    t0 = time.time()
    with torch.no_grad():
        output = net(input_batch, w=w, adain=True)[0]
        for i, iter in enumerate(output):
            restored_face = tensor2img(iter, rgb2bgr=True, min_max=(-1, 1)).astype('uint8')
            face_helper.add_restored_face(restored_face, face_helper.cropped_faces[i])

    # face_helper.get_inverse_affine(None)
    face_helper.paste_faces_to_input_image(upsample_img=None, draw_box=False)

    del output
    torch.cuda.empty_cache()

    t1 = time.time()
    print("Enhancing took: ", t1 - t0)

    return face_helper.restored_faces
    # show result
    # for restored_face in restored_faces:
    #     Image.fromarray(restored_face[:, :, ::-1]).show()
