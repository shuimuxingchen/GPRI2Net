import cv2 as cv
import numpy as np


def normalize(pic):
    float_pic = pic.astype("float32")
    return float_pic / 255

if __name__ == "__main__":
    ar = np.array([[[
        [1,2,3],
        [4,5,6]
    ]]], "uint8")
    nar = normalize(ar)