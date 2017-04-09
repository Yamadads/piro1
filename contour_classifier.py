import image_utils as iu
import cv2
import numpy as np


def get_classification(normalized_figures):
    classification = []
    print get_junction_contour(normalized_figures[0])

    for i in range(len(normalized_figures)):
        classification.append(i)

    return classification


def get_junction_contour(figure):

    contour = cv2.approxPolyDP(iu.get_contour(figure), 2, True)

    max_y = max(contour[:, 0, 1])
    threshold = 5

    result = sorted([r.tolist() for r in contour[:,0,:] if r[1] < max_y - threshold], key=lambda idx: idx[0])

    return result
