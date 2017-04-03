import cv2
import numpy as np


def load_pictures(directory_path, pictures_no):
    pictures = []
    for i in range(pictures_no):
        img_path = directory_path + "\\" + str(i) + ".png"
        image = cv2.imread(img_path, 0)
        pictures.append(image)
    return pictures


def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_contour(image):
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    return contours[0]


def get_normalized_figure(img, normalized_width):
    contour = get_contour(img)
    box = get_box(contour)
    angle = get_angle(box[0], box[1])
    rotated_img = rotate_image(img, angle)
    rotated_contour = get_contour(rotated_img)
    box2 = get_box(rotated_contour)
    cropped_image = rotated_img[box2[1][1]:box2[0][1], box2[0][0]:box2[2][0]]
    scale = normalized_width / len(cropped_image[0])
    scaled_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
    show_image(scaled_image)


def get_box(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 0.5)
    result = cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_CUBIC)
    return result


def get_angle(p0, p1):
    p2 = []
    p2.append(p1[0])
    p2.append(p0[1])

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(-angle)
