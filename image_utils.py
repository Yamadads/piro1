import cv2
import numpy as np
import sys


def load_pictures(directory_path, pictures_no):
    pictures = []
    for i in range(pictures_no):
        img_path = directory_path + "\\" + str(i) + ".png"
        image = cv2.imread(img_path, 0)
        pictures.append(image)
    return pictures


def show_image(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_contour(image):
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    kernel = np.ones((4, 4), np.uint8)
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    _, contours, hierarchy = cv2.findContours(closed_thresh, 1, 2)
    return get_contour_with_max_size(contours)


def get_contour_with_max_size(contours):
    if len(contours) > 1:
        max_length_contour = 0
        max_length = 0
        for i in contours:
            if len(i) > max_length:
                max_length_contour = i
                max_length = len(i)
        return max_length_contour
    return contours[0]


def get_normalized_figure(img, normalized_width, text):
    contour = get_contour(img)
    box = get_box(contour)
    angle = get_angle(box[0], box[1])
    rotated_img = rotate_bound(img, angle)
    rotated_contour = get_contour(rotated_img)
    box2 = get_box(rotated_contour)
    cropped_image = get_cropped_image(rotated_img, box2)
    scale = float(normalized_width) / float(len(cropped_image[0]))
    scaled_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
    final_image = final_rotate(scaled_image, normalized_width)
    show_image(text, final_image)


def final_rotate(image, normalized_width):
    images = []
    images.append(image)
    #print (images[0][:][-10:])
    for i in range(3):
        images.append(rotate_bound(images[i], 90))
        scale = float(normalized_width) / float(len(images[i + 1][0]))
        images[i + 1] = cv2.resize(images[i + 1], (0, 0), fx=scale, fy=scale)
        #print (images[i+1][:][-10:])
        #show_image(str(i),images[i+1])
    max_base = 0
    base = 0
    for i in range(4):
        base_sum = np.sum(images[i][:][-30:-2])
        if base_sum>max_base:
            max_base = base_sum
            base = i
    return images[base]


def get_cropped_image(image, box):
    min_x = min_y = sys.maxint
    max_x = max_y = 0
    for i in box:
        if i[0] > max_x:
            max_x = i[0]
        if i[0] < min_x:
            min_x = i[0]
        if i[1] > max_y:
            max_y = i[1]
        if i[1] < min_y:
            min_y = i[1]
    cropped_image = image[min_y:max_y, min_x:max_x]
    return cropped_image


def get_box(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def get_angle(p0, p1):
    p2 = []
    p2.append(p1[0])
    p2.append(p0[1])

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)


def rotate_bound(image, angle):
    (height, width) = image.shape[:2]
    (center_X, center_Y) = (width // 2, height // 2)

    rot = cv2.getRotationMatrix2D((center_X, center_Y), -angle, 1.0)
    cos = np.abs(rot[0, 0])
    sin = np.abs(rot[0, 1])

    new_Width = int((height * sin) + (width * cos))
    new_Height = int((height * cos) + (width * sin))

    rot[0, 2] += (new_Width / 2) - center_X
    rot[1, 2] += (new_Height / 2) - center_Y

    return cv2.warpAffine(image, rot, (new_Width, new_Height))
