import cv2
import numpy as np
import matplotlib

def test_opencv():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert img.shape == (100, 100, 3)

def test_numpy():
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6

def test_matplotlib():
    assert matplotlib.__version__ is not None