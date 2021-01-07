import numpy as np
import math

sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]


def cross_correlation_1d(img, kernel):
    img_height = img.shape[0]
    img_width = img.shape[1]
    kernel_size = kernel.size
    result_arr = np.empty(img.shape, dtype=np.float64)

    # horizontal kernel
    if kernel.ndim == 1:

        # ------------------------------padding------------------------------------ #
        column_padding = (kernel_size - 1) / 2

        while column_padding > 0:
            img = np.insert(img, 0, values=img[:, 0], axis=1)
            img = np.insert(img, img_width + 1, values=img[:, img_width], axis=1)
            img_width = img.shape[1]
            column_padding -= 1
        # -------------------------------------------------------------------------- #

        fimg_width = img_width - kernel_size + 1
        fimg_height = img_height

        for i in range(0, fimg_height):
            for j in range(0, fimg_width):
                target = img[i:i + 1, j:j + kernel_size]
                result = np.multiply(target, kernel)
                result_arr[i][j] = result.sum()

        return result_arr

    # vertical kernel
    else:

        # ------------------------------padding------------------------------------ #

        row_padding = (kernel_size - 1) / 2

        while row_padding > 0:
            img = np.insert(img, 0, values=img[0, :], axis=0)
            img = np.insert(img, img_height + 1, values=img[img_height, :], axis=0)
            img_height = img.shape[0]
            row_padding -= 1

        # -------------------------------------------------------------------------- #

        fimg_width = img_width
        fimg_height = img_height - kernel_size + 1

        for i in range(0, fimg_height):
            for j in range(0, fimg_width):
                target = img[i:i + kernel_size, j:j + 1]
                result = np.multiply(target, kernel)
                result_arr[i][j] = result.sum()

        return result_arr


def cross_correlation_2d(img, kernel):
    img_height = img.shape[0]
    img_width = img.shape[1]
    kernel_size = len(kernel)
    result_arr = np.empty(img.shape, dtype=np.float64)

    # ------------------------------padding------------------------------------ #
    column_padding = (kernel_size - 1) / 2
    row_padding = (kernel_size - 1) / 2

    while column_padding > 0:
        img = np.insert(img, 0, values=img[:, 0], axis=1)
        img = np.insert(img, img_width + 1, values=img[:, img_width], axis=1)
        img_width = img.shape[1]
        column_padding -= 1

    while row_padding > 0:
        img = np.insert(img, 0, values=img[0, :], axis=0)
        img = np.insert(img, img_height + 1, values=img[img_height, :], axis=0)
        img_height = img.shape[0]
        row_padding -= 1

    # -------------------------------------------------------------------------- #

    fimg_width = img_width - kernel_size + 1
    fimg_height = img_height - kernel_size + 1

    for i in range(0, fimg_height):
        for j in range(0, fimg_width):
            target = img[i:i + kernel_size, j:j + kernel_size]
            result = np.multiply(target, kernel)
            result_arr[i][j] = result.sum()

    return result_arr


def get_gaussian_filter_1d(size, sigma):
    kernel = [0.0] * size
    i = 0
    for j in range(0, size):
        kernel[j] = float(math.exp(-(((i - int(size / 2)) ** 2 + (j - int(size / 2)) ** 2) / (2 * sigma ** 2)))) \
                    / float(2 * math.pi * sigma ** 2)

    kernel = kernel / np.sum(kernel)
    return np.array(kernel)


def get_gaussian_filter_2d(size, sigma):
    kernel = [[0] * size for _ in range(size)]

    for i in range(0, size):
        for j in range(0, size):
            kernel[i][j] = float(math.exp(-(((i - int(size / 2)) ** 2 + (j - int(size / 2)) ** 2) / (2 * sigma ** 2)))) \
                           / float(2 * math.pi * sigma ** 2)

    kernel = kernel / np.sum(kernel)
    return np.array(kernel)
