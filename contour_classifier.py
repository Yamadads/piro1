from pandas.core.nanops import disallow

import image_utils as iu
import cv2
import numpy as np
import copy

hu_moments_in_use = 6


def get_classification(normalized_figures):

    hu_moments = [[] for i in range(0, hu_moments_in_use)]

    for image_i in range(0, len(normalized_figures)):

        junction0 = extract_junction_image(normalized_figures[image_i])

        ret, junction0 = cv2.threshold(junction0, 127 , 255, 0)
        # iu.show_image('2', junction0)

        #obliczamy 1 i 2 moment Hu
        insert_hu_moments(junction0, hu_moments, image_i, False)

        # junction_1 = iu.rotate_bound(np.invert(junction0), 180)
        junction_1 = np.invert(junction0)

        # obraz sie rozmazuje... wiec go binaruzejmy


        # obliczamy jego Hu moments
        insert_hu_moments(junction_1, hu_moments, image_i, True)


    for hu_moment_id in range(0, hu_moments_in_use):
        hu_moments[hu_moment_id].sort(key=lambda x:x[0])

    # calculate every one to very one distance on every hu moment
    dist = {}
    for hu_moment_id in range(0, hu_moments_in_use):
        for i in range(0, len(hu_moments[hu_moment_id]) - 1):
            for el_i in range(i + 1, len(hu_moments[hu_moment_id])):
                # can only compare inverted with non inverted images
                if hu_moments[hu_moment_id][i][2] == hu_moments[hu_moment_id][el_i][2]:
                    continue

                if hu_moments[hu_moment_id][i][1] == hu_moments[hu_moment_id][el_i][1]:
                    continue

                tuple_pair = sorted([hu_moments[hu_moment_id][i][1], hu_moments[hu_moment_id][el_i][1]])
                tuple_id = str(tuple_pair)[1:-1]

                distance = hu_moments[hu_moment_id][el_i][0] - hu_moments[hu_moment_id][i][0]

                if tuple_id not in dist:
                    # print 'id[{0}] i={1} el_i={2}'.format(tuple_id, i, el_i)
                    dist[tuple_id] = [0, tuple_pair]

                dist[tuple_id][0] += distance


    sorted_dist = sorted(dist.items(), key=lambda x:x[1], reverse=False)

    sorted_dist_orig = copy.copy(sorted_dist)
    #for val in sorted_dist:
    #    print val

    results = []

    while True:

        best = sorted_dist[0]

        sorted_dist = [x for x in sorted_dist if not partly_same_tuple(x[1][1], best[1][1])]

        results.append(best[1][1])

        if len(sorted_dist) == 0:
            break

    ordered_result = [[] for i in range(len(normalized_figures))]

    for x in results:

        ordered_result[x[0]].append(x[1])
        ordered_result[x[1]].append(x[0])

        append_rest(x[0], x, ordered_result[x[0]], sorted_dist_orig)
        append_rest(x[1], x, ordered_result[x[1]], sorted_dist_orig)

    return ordered_result


def append_rest(target, tuple, result, sorted_dist_orig):
    counter = 4
    for next_cand in sorted_dist_orig:
        if next_cand[1][1] == tuple:
            continue

        if target in next_cand[1][1]:
            counter -= 1

            result.append(next_cand[1][1][0] if next_cand[1][1][1] == target else next_cand[1][1][1])
            if counter == 0:
                break


def partly_same_tuple(t_a, t_b):
    for x in [t_b[0], t_b[1]]:
        if x in [t_a[0], t_a[1]]:
            return True
    return False

def insert_hu_moments(image, hu_list, id, reversed):
    # obliczamy momenty Hu
    hu_jun_0 = cv2.HuMoments(cv2.moments(image)).flatten()

    # tak wyglada -np.sign(hu_jun_0) * np.log10(np.abs(hu_jun_0)) normalizacja momentow hu
    # ale w naszym przypadku nie potrzebujemy chyba az takiej informacji
    log_hu_jun_0 = -np.sign(hu_jun_0) * np.log10(np.abs(hu_jun_0))

    # print log_hu_jun_0

    # odkladamy pierwszy i drugi moment Hu do tabeli z wartosciamy dla wszystkich figur
    for hu_moment_id in range(0, hu_moments_in_use):
        hu_list[hu_moment_id].append([log_hu_jun_0[hu_moment_id], id, reversed])


def extract_junction_image(image):
    contour = get_junction_contour(image)

    max_y = max(contour, key=lambda x:x[1])[1]

    return image[:][:min((max_y + 1), len(image[0]))]

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
