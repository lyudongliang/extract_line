import cv2 as cv
import numpy as np
import os


def get_sorted_contours(contours):
    sorted_contours = sorted(contours, key=lambda x: -x.shape[0])

    return sorted_contours


def get_contours(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # binary_image = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    cv.imshow('binary_image', binary_image)
    cv.waitKey(0)

    contours, hierarchy = cv.findContours(binary_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # cv.drawContours(image, contours, -1, (0, 0, 255), 2)
    # new_contours = contours[:10000]
    # cv.drawContours(image, new_contours, -1, (0, 0, 255), 2)

    sorted_contours = get_sorted_contours(contours)

    new_contours = sorted_contours[:10]
    cv.drawContours(image, new_contours, -1, (255, 0, 0), 2)

    # x, y, w, h = cv.boundingRect(new_contours[0])
    # image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for i in range(10):
        rect = cv.minAreaRect(new_contours[i])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        image = cv.drawContours(image, [box], 0, (0, 0, 255), 2)

    cv.imshow('walk_corner_contours', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return image


picture_types = ['walk', 'road', 'niu', 'home', 'table', 'window', 'office']
type = picture_types[4]

picture_file = type + '_corner.jpg'
picture_directory = 'D:\data\picture'

src_image = cv.imread(os.path.join(picture_directory, picture_file))

image_contours = get_contours(src_image)

output_directory = os.path.join('picture', type + '_contours.jpg')

cv.imwrite(output_directory, image_contours)
