from pandas.core.nanops import disallow

import image_utils as iu
import cv2
import numpy as np


def get_classification(normalized_figures):
    hu_moments = [[], []]

    for image_i in range(0, len(normalized_figures)):

        junction0 = extract_junction_image(normalized_figures[image_i])
        #obliczamy 1 i 2 moment Hu
        insert_hu_moments(junction0, hu_moments, image_i)

        junction_1 = iu.rotate_bound(np.invert(junction0), 180)
        # obraz sie rozmazuje... wiec go binaruzejmy
        ret, junction_1 = cv2.threshold(junction_1, 127, 255, 0)
        # obliczamy jego Hu moments
        insert_hu_moments(junction_1, hu_moments, image_i)


    hu_moments[0].sort(key=lambda x:x[0])
    hu_moments[1].sort(key=lambda x:x[0])

    results = [];

    dist = [[], []]

    for i_m in range(0, 2):
        dist[i_m] = [
            [hu_moments[i_m][i + 1][0] - hu_moments[i_m][i][0], [hu_moments[i_m][i][1], hu_moments[i_m][i + 1][1]]] for
            i in range(0, len(hu_moments[i_m]) - 1)]
        sorted(dist[i_m], reverse=True)
        # dist[i_m].sort(key=lambda x:x[0], reversed=True)

    while True:

        bests = [dist[0][0], dist[1][0]]

        id_smaller = 0 if bests[0][0] < bests[1][0] else 1
        id_bigger = abs(id_smaller - 1)

        # remove from smaller
        dist[id_smaller]= [x for x in dist[id_smaller] if not partly_same_tuple(x[1], bests[id_smaller][1])]


        # remove leftout from bigger
        dist[id_bigger] = [x for x in dist[id_bigger] if not partly_same_tuple(x[1], bests[id_bigger][1])]

        results.append(bests[id_smaller][1])

        # print results

        if len(dist[0]) == 0:
            break

    ordered_result = np.zeros(len(results) * 2)

    for x in results:
        ordered_result[x[0]] = x[1]
        ordered_result[x[1]] = x[0]

    return ordered_result


def partly_same_tuple(t_a, t_b):
    for x in [t_b[0], t_b[1]]:
        if x in [t_a[0], t_a[1]]:
            return True
    return False


def insert_hu_moments(image, hu_list, id):
    # obliczamy momenty Hu
    hu_jun_0 = cv2.HuMoments(cv2.moments(image)).flatten()

    # tak wyglada -np.sign(hu_jun_0) * np.log10(np.abs(hu_jun_0)) normalizacja momentow hu
    # ale w naszym przypadku nie potrzebujemy chyba az takiej informacji
    log_hu_jun_0 = hu_jun_0 # -np.sign(hu_jun_0) * np.log10(np.abs(hu_jun_0))

    # odkladamy pierwszy i drugi moment Hu do tabeli z wartosciamy dla wszystkich figur
    hu_list[0].append([log_hu_jun_0[0], id])
    hu_list[1].append([log_hu_jun_0[1], id])


def extract_junction_image(image):
    contour = get_junction_contour(image)

    max_y = max(contour, key=lambda x:x[1])[1]

    return image[:][:max(1, max_y)]

    # slice = [arr[i][0:2] for i in range(0, 2)]

    # new_image = [image[i][0:max_y] for i in range(0, len(image))]
    # iu.show_image('dsd', image)
    # iu.show_image('dsd', new_image)


def get_junction_contour(figure):

    contour = cv2.approxPolyDP(iu.get_contour(figure), 2, True)

    max_y = max(contour[:, 0, 1])
    threshold = 5

    result = sorted([r.tolist() for r in contour[:, 0, :] if r[1] < max_y - threshold], key=lambda idx: idx[0])

    return result
