import cv2
import image_utils as iu
import numpy as np


def get_classification(normalized_figures):
    classification = []
    junction_images = get_junction_images(normalized_figures)
    for i in range(0, len(normalized_figures)):
        difference = {}
        if junction_images[i] == None:
            classification.append([1,2,3,4,5])
            continue
        rotated_junction_image = iu.rotate_bound(junction_images[i], 180)
        for j in range(0, len(normalized_figures)):
            if i != j:
                if junction_images[j] == None:
                    difference[j]=0
                    continue
                resize_param = float(len(rotated_junction_image))/float(len(junction_images[j]))
                test_image = cv2.resize(junction_images[j], None, fx=1, fy=resize_param, interpolation = cv2.INTER_CUBIC)
                diff = rotated_junction_image ^ test_image
                difference[j]=np.sum(diff[:])
                if difference[j]>(0.93*len(rotated_junction_image)*len(rotated_junction_image[0])*255):
                    break
        classification.append(sorted(difference.keys(), key=difference.get, reverse=True)[:5])
    return classification


def get_junction_images(images):
    junction_images = []
    for image_i in range(0, len(images)):
        junction_image = extract_junction_image(images[image_i])
        junction_images.append(iu.get_binary_image(junction_image))
    return junction_images


def extract_junction_image(image):
    contour = get_junction_contour(image)
    max_y = max(contour, key=lambda x:x[1])[1]
    if (max_y<10):
        max_y = 10
    return image[:][:max_y]


def get_junction_contour(figure):

    contour = cv2.approxPolyDP(iu.get_contour(figure), 7, True)
    #print(len(contour))
    #print(len(contour[0]))
    #print("___")

    max_y = max(contour[:, 0, 1])
    threshold = 20

    result = sorted([r.tolist() for r in contour[:, 0, :] if r[1] < max_y - threshold], key=lambda idx: idx[0])

    return result