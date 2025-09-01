from functools import lru_cache
from typing import List, Dict, Any
import numpy as np

class _EasyOCRWrapper:
    def __init__(self):
        # Lazy import to speed app startup
        import easyocr  # type: ignore
        # CPU-only, English
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def ocr(self, img_rgb: np.ndarray) -> List[Dict[str, Any]]:
        # EasyOCR expects RGB ndarray
        results = self.reader.readtext(img_rgb, detail=1, paragraph=False)
        lines: List[Dict[str, Any]] = []
        for box, text, conf in results:
            # box is a list of 4 points; keep as-is for display/debugging
            lines.append({"text": (text or "").strip(), "conf": float(conf), "box": box})
        return lines

@lru_cache(maxsize=1)
def get_ocr():
    # Cache the reader to avoid reloading models
    return _EasyOCRWrapper()

def run_ocr(wrapper: _EasyOCRWrapper, image_bgr_or_gray: np.ndarray) -> List[Dict[str, Any]]:
    import cv2
    # Convert input to RGB for EasyOCR
    if len(image_bgr_or_gray.shape) == 2:
        rgb = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2RGB)
    return wrapper.ocr(rgb)
