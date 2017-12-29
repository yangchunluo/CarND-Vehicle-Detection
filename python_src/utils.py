import os
import cv2
import numpy as np


def get_img_size(img):
    """
    Get image size
    :param img: image pixels array
    :return: a tuple of width (X) and height (Y)
    """
    return img.shape[1], img.shape[0]


def output_img(img, path):
    """
    Write image as an output file.
    :param img: image pixels array, in RGB color space or gray scale
    :param path: output file path
    """
    # Recursively creating the directories leading to this path
    dirs = [path]
    for _ in range(2):
        dirs.append(os.path.dirname(dirs[-1]))
    for d in dirs[:0:-1]:
        if d and not os.path.exists(d):
            os.mkdir(d)
    # If color image, convert to BGR to write (cv2.imwrite takes BGR image).
    # Otherwise it is gray scale.
    if len(img.shape) == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def insert_image(canvas, insert, x, y, shrinkage=None):
    """
    Overlay a small image on a background image as an insert.
    :param canvas: background image
    :param insert: inserted image
    :param x: X position
    :param y: Y position
    :param shrinkage: optional shrinkage factor
    """
    insert_size = get_img_size(insert)
    if shrinkage is not None:
        insert = cv2.resize(insert, (int(insert_size[0] / shrinkage),
                                     int(insert_size[1] / shrinkage)))
        insert_size = get_img_size(insert)
    x = int(x)
    y = int(y)
    if len(insert.shape) < 3:
        insert = cv2.cvtColor(insert.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    canvas[y:y + insert_size[1], x:x + insert_size[0]] = insert