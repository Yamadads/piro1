import image_utils as iu
import cv2

def get_classification(normalized_figures):
    classification = []
    get_contour(normalized_figures[0])
    for i in range(len(normalized_figures)):
        classification.append(i)
    return classification

def get_contour(figure):
    cont = iu.get_contour(figure)
    cont = cv2.approxPolyDP(cont, 2,True)
    print (cont)
