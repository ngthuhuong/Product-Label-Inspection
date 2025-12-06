import cv2
import numpy as np
import config

def sobel_edges(gray, ksize=3, thresh=None):
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    mag = mag / (mag.max() + 1e-6)
    mag = (mag * 255).astype(np.uint8)

    if thresh is not None:
        _, mag = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)

    return mag

def build_label_mask_sobel(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_roi, config.GAUSSIAN_BLUR_KERNEL, 0)
    edges = sobel_edges(blur, ksize=3, thresh=40)
    return edges


def prewitt_edges(gray, thresh=None):
    kx = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    ky = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)

    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)

    mag = cv2.magnitude(gx, gy)

    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag = mag.astype(np.uint8)

    _, edges = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)
    return edges

def build_label_mask_prewitt(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_roi, config.GAUSSIAN_BLUR_KERNEL, 0)
    edges = prewitt_edges(blur, thresh=40)
    kernel_lbl = np.ones((config.KERNEL_LABEL_SIZE, config.KERNEL_LABEL_SIZE), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_lbl,
                             iterations=config.LABEL_CLOSE_ITER)
    return edges

def canny_edges(gray, low=50, high=150):
    edges = cv2.Canny(gray, low, high)
    return edges


def build_label_mask_canny(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_roi, config.GAUSSIAN_BLUR_KERNEL, 0)
    edges = cv2.Canny(blur, 80, 200)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
    return edges



