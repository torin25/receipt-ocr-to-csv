from typing import Tuple
import cv2
import numpy as np
from PIL import Image

def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def _denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)

def _binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)

def _deskew(binary: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(binary == 0))
    if coords.size == 0:
        return binary
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    (h, w) = binary.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_image(pil: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = _to_gray(bgr)
    den = _denoise(gray)
    bin_img = _binarize(den)
    desk = _deskew(bin_img)
    disp_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return desk, disp_rgb
