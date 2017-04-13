import image_utils as iu
import cv2
import numpy as np
import operator

vertices = {}


def get_classification(normalized_figures):
    classification = {}
    for i in range(len(normalized_figures)):
        contour = get_junction_contour(normalized_figures[i])
        vertices[i]=(len(contour))

    for i in range(len(normalized_figures)):
        x = {}
        for j in range(len(normalized_figures)):
            if i!=j:
                x[j]=abs(vertices[i]-vertices[j])
        sorted_x=sorted(x.items(),key=lambda x: x[1])
        classification[i] = [l[0] for l in sorted_x]
    return classification


def get_junction_contour(figure):

    contour = cv2.approxPolyDP(iu.get_contour(figure), 2, True)

    max_y = max(contour[:, 0, 1])
    threshold = 5

    result = sorted([r.tolist() for r in contour[:,0,:] if r[1] < max_y - threshold], key=lambda idx: idx[0])

    return result
