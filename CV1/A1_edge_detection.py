import cv2
import numpy as np
import my_functions as my
import time


def compute_image_gradient(img):
    # convert to cross-correlation filter
    sobel_x_cc = np.array(np.flip(my.sobel_x))
    sobel_y_cc = np.array(np.flip(my.sobel_y))

    dx = my.cross_correlation_2d(img, sobel_x_cc)
    dy = my.cross_correlation_2d(img, sobel_y_cc)

    magnitude = np.sqrt(np.square(dx) + np.square(dy))
    magnitude *= 255.0 / np.max(magnitude)
    magnitude = magnitude.astype(np.uint8)

    direction = np.rad2deg(np.arctan2(dy, dx))
    direction[direction < 0] += 360

    return direction, magnitude


def non_maximum_suppression_dir(mag, dir):
    result_mag = np.zeros(mag.shape)
    height = mag.shape[0]
    width = mag.shape[1]
    degree_list = [0, 45, 90, 135, 180, 225, 270, 315]

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            target = mag[i][j]
            quan_dir = find_nearest(degree_list, dir[i][j])

            if quan_dir == 0 or quan_dir == 180:
                if mag[i][j + 1] <= target and mag[i][j - 1] <= target:
                    result_mag[i][j] = mag[i][j]

            elif quan_dir == 45 or quan_dir == 225:
                if mag[i - 1][j - 1] <= target and mag[i + 1][j + 1] <= target:
                    result_mag[i][j] = mag[i][j]

            elif quan_dir == 90 or quan_dir == 270:
                if mag[i - 1][j] <= target and mag[i + 1][j] <= target:
                    result_mag[i][j] = mag[i][j]

            elif quan_dir == 135 or quan_dir == 315:
                if mag[i + 1][j - 1] <= target and mag[i - 1][j + 1] <= target:
                    result_mag[i][j] = mag[i][j]

    result_mag = result_mag.astype(np.uint8)

    return result_mag


def find_nearest(array, value):
    array = np.asarray(array)
    if value > 337.5:
        return array[0]
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def main():

    for target in ['shapes.png', 'lenna.png']:
        print("-------------------------------------------------------")
        print("<", target, ">\n")
        img = cv2.imread(target, cv2.IMREAD_GRAYSCALE)

        kernel = my.get_gaussian_filter_2d(7, 1.5)
        fimg = my.cross_correlation_2d(img, kernel)

        start = time.time()
        direction, raw_mag = compute_image_gradient(fimg)
        print("compute_image_gradient time:", time.time() - start)

        cv2.imwrite('./result/part_2_edge_raw_' + target, raw_mag)
        cv2.imshow('RAW_MAG', raw_mag)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        start = time.time()
        suppressed_mag = non_maximum_suppression_dir(raw_mag, direction)
        print("non_maximum_suppression_dir time: ", time.time() - start)

        cv2.imwrite('./result/part_2_edge_sup_' + target, suppressed_mag)
        cv2.imshow('SUPPRESSED_MAG', suppressed_mag)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
