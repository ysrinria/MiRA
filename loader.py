# loader.py
import os
import cv2
import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from petrel_client.client import Client
    petrel_backend_imported = True
except ImportError:
    petrel_backend_imported = False


def get_video_loader(use_petrel_backend: bool = True,
                     enable_mc: bool = True,
                     conf_path: str = None):
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    def _loader(video_path):
        if _client is not None and 's3:' in video_path:
            video_path = io.BytesIO(_client.get(video_path))

        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        return vr

    return _loader

    
def get_image_loader(use_petrel_backend: bool = True,
                     enable_mc: bool = True,
                     conf_path: str = None):
    """원래 JPG 로더 (S3/Petrel + OpenCV). RGB/uint8/HWC 반환"""
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    def _loader(frame_path):
        if _client is not None and 's3:' in frame_path:
            img_bytes = _client.get(frame_path)
        else:
            with open(frame_path, 'rb') as f:
                img_bytes = f.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  # BGR, uint8
        if img is None:
            return None
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)     # in-place → RGB
        return img
    return _loader


def get_image_png_loader():
    """PNG 전용(Pillow). RGB/uint8/HWC 반환"""
    def _loader(path):
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))
    return _loader


def get_image_smart_loader(use_petrel_backend: bool = True,
                           enable_mc: bool = True,
                           conf_path: str = None):
    """확장자 기준 자동 분기: JPG/JPEG→OpenCV, 그 외(또는 실패)→Pillow"""
    jpg_loader = get_image_loader(use_petrel_backend, enable_mc, conf_path)
    png_loader = get_image_png_loader()

    def _loader(path):
        ext = os.path.splitext(path)[-1].lower()
        if ext in (".jpg", ".jpeg"):
            img = jpg_loader(path)
            if img is not None:
                return img
        return png_loader(path)
    return _loader