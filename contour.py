import cv2 as cv
import numpy as np
import os


def get_contours(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow('binary_image', binary_image)
    # cv.waitKey(0)

    contours, hierarchy = cv.findContours(binary_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # cv.drawContours(image, contours, -1, (0, 0, 255), 2)
    # new_contours = contours[:10000]
    # cv.drawContours(image, new_contours, -1, (0, 0, 255), 2)

    sorted_contours = get_sorted_contours(contours)

    new_contours = sorted_contours[:10]
    cv.drawContours(image, new_contours, -1, (255, 0, 0), 2)

    cv.imshow('walk_corner_contours', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return image


def get_sorted_contours(contours):
    sorted_contours = sorted(contours, key=lambda x: -x.shape[0])

    return sorted_contours


picture_types = ['walk', 'road', 'niu', 'home', 'table', 'window', 'office']
type = picture_types[0]

picture_file = type + '_corner.jpg'
picture_directory = 'D:\data\picture'

src_image = cv.imread(os.path.join(picture_directory, picture_file))

image_contours = get_contours(src_image)

output_dirctory = os.path.join('picture', type + '_contours.jpg')

cv.imwrite(output_dirctory, image_contours)
