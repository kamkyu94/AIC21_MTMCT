import math
import cv2
import config
import random
import numpy as np


# Resize a rectangular image to a padded rectangular
def letterbox(img, color=(0, 0, 0)):
    # shape = [height, width]
    shape = img.shape[:2]
    ratio = min(float(config.img_h) / shape[0], float(config.img_w) / shape[1])

    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))

    # Padding
    dh = (config.img_h - new_shape[1]) / 2
    dw = (config.img_w - new_shape[0]) / 2

    # Top, bottom, left, right
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # resized, no border, padded rectangular
    image = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, ratio, left, top


def random_affine(img, degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-2, 2), borderValue=(0, 0, 0)):

    # width of added border (optional)
    border = 0
    height, width = img.shape[0], img.shape[1]

    # Rotation
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]

    # Scale
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    M = S @ T @ R

    # BGR order borderValue
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,  borderValue=borderValue)

    return imw
