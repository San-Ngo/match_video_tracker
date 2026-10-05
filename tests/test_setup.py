import cv2
import numpy as np


def test_numpy_and_opencv_work():
    img = np.zeros((10, 20, 3), dtype=np.uint8)   # a tiny black image: 10 tall, 20 wide
    assert img.shape == (10, 20, 3)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # colour -> grayscale drops the 3rd axis
    assert gray.shape == (10, 20)
