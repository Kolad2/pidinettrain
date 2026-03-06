import numpy as np

_debug = True

def uint8_normalize(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        if _debug:
            print("Normalization passed")
        return img
    if np.issubdtype(img.dtype, np.floating):
        if _debug:
            print("Floating normalization")
        return _float_to_uint8(img)
    if img.dtype == np.uint16:
        if _debug:
            print("unit16 normalization")
        return _uint16_to_uint8(img)
    raise ValueError(f"Unsupported image dtype: {img.dtype}")

def _uint16_to_uint8(img: np.ndarray) -> np.ndarray:
    return (img >> 8).astype(np.uint8)

def _float_to_uint8(img: np.ndarray) -> np.ndarray:
    v_min = np.nanmin(img)
    v_max = np.nanmax(img)
    if v_min < 0:
        raise ValueError("img has negative values")
    if v_max <= 0:
        raise ValueError("img is fully nonpositive values")
    scale = 255/v_max
    np.multiply(img, scale, out=img)
    return img.astype(np.uint8)

