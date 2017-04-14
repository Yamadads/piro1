import image_utils as iu
import cv2
import numpy as np


def get_classification(normalized_figures):
    classification = []

    # test(normalized_figures)
    # return

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

    dist = [[],[]]

    for i_m in range(0, 1):
        dist[i_m] = [[hu_moments[i_m][i + 1][0] - hu_moments[i_m][i][0], [hu_moments[i_m][i][1], hu_moments[i_m][i + 1][1]]] for i in range(0, len(hu_moments[i_m]) - 1)]
        dist[i_m].sort(key=lambda x:x[0])

    #print dist

    #print hu_moments[0]
    #print hu_moments[1]

    print '=================='

    #print dist

    # TODO

    return


def test(normalized_figures):

    img_test0 = normalized_figures[0]
    img_test1 = normalized_figures[5]

    junction0 = extract_junction_image(img_test0)
    junction1 = extract_junction_image(img_test1)

    # iu.show_image('junction1', junction1)
    rij1 = iu.rotate_bound(np.invert(junction1), 180)

    # print cv2.HuMoments(cv2.moments(rij1)).flatten()
    # iu.show_image('inverted rotaded j5', rij1)


    ret, tresh_junction_1 = cv2.threshold(rij1, 127, 255, 0)
    hu_jun_1 = cv2.HuMoments(cv2.moments(tresh_junction_1)).flatten()

    hu_jun_0 = cv2.HuMoments(cv2.moments(junction0)).flatten()

    print hu_jun_0
    print hu_jun_1

    print ('============')
    print -np.sign(hu_jun_1) * np.log10(np.abs(hu_jun_1))
    print -np.sign(hu_jun_0) * np.log10(np.abs(hu_jun_0))

    return


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

    return image[:][:max_y]

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
