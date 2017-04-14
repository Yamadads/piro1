import cv2
import image_utils as iu
import numpy as np


def get_classification(normalized_figures):
    classification = []
    junction_images = get_junction_images(normalized_figures)
    for i in range(0, len(normalized_figures)):
        difference = {}
        rotated_junction_image = iu.rotate_bound(junction_images[i], 180) #np.rot90(junction_images[i])#
        for j in range(0, len(normalized_figures)):
            if i != j:
                resize_param = float(len(rotated_junction_image))/float(len(junction_images[j]))
                test_image = cv2.resize(junction_images[j], None, fx=1, fy=resize_param, interpolation = cv2.INTER_CUBIC)
                diff=rotated_junction_image^test_image
                difference[j]=np.sum(diff[:])
        classification.append(sorted(difference.keys(), key=difference.get, reverse=True))
    return classification


def get_junction_images(images):
    junction_images = []
    for image_i in range(0, len(images)):
        # iu.show_image("test", normalized_figures[image_i], 1000)
        junction_image = extract_junction_image(images[image_i])
        junction_images.append(junction_image)
    return junction_images


def extract_junction_image(image):
    contour = get_junction_contour(image)

    max_y = max(contour, key=lambda x:x[1])[1]

    return image[:][:max_y]


def get_junction_contour(figure):

    contour = cv2.approxPolyDP(iu.get_contour(figure), 2, True)

    max_y = max(contour[:, 0, 1])
    threshold = 5

    result = sorted([r.tolist() for r in contour[:, 0, :] if r[1] < max_y - threshold], key=lambda idx: idx[0])

    return result