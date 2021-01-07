import cv2
import numpy as np
import my_functions as my
import time


def compute_corner_response(img):
    # convert to cross-correlation filter
    sobel_x_cc = np.array(np.flip(my.sobel_x))
    sobel_y_cc = np.array(np.flip(my.sobel_y))

    sobelx = my.cross_correlation_2d(img, sobel_x_cc)
    sobely = my.cross_correlation_2d(img, sobel_y_cc)

    IxIx = np.multiply(sobelx, sobelx)
    IxIy = np.multiply(sobelx, sobely)
    IyIy = np.multiply(sobely, sobely)

    window_size = 5
    fimg_width = img.shape[1] - window_size + 1
    fimg_height = img.shape[0] - window_size + 1
    empt = int((window_size - 1) / 2)

    response = np.zeros(img.shape)

    for x in range(0, fimg_height):
        for y in range(0, fimg_width):
            m_xx = IxIx[x:x + window_size, y:y + window_size].sum()
            m_xy = IxIy[x:x + window_size, y:y + window_size].sum()
            m_yy = IyIy[x:x + window_size, y:y + window_size].sum()

            det = m_xx * m_yy - m_xy * m_xy
            trace = m_xx + m_yy

            R = det - 0.04 * trace ** 2
            if R < 0: R = 0  # update all the negative responses to 0

            response[x+empt][y+empt] = R

    # normalize responses to a range of [0, 1]
    cv2.normalize(response, response, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    return response


def non_maximum_suppression_win(R, winSize):
    suppressed_R = np.zeros(R.shape)
    empt = int((winSize - 1) / 2)
    fheight = R.shape[0] - empt
    fwidth = R.shape[1] - empt

    for i in range(empt, fheight):
        for j in range(empt, fwidth):
            target = R[i][j]
            if target <= 0.1:
                continue
            box = R[i - empt:i + empt + 1, j - empt:j + empt + 1]
            if target == np.max(box):
                suppressed_R[i][j] = R[i][j]

    return suppressed_R


def main():
    for target in ['shapes.png', 'lenna.png']:
        img = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
        print("-------------------------------------------------------")
        print("<", target, ">\n")

        img_height = img.shape[0]
        img_width = img.shape[1]

        kernel = my.get_gaussian_filter_2d(7, 1.5)
        fimg = my.cross_correlation_2d(img, kernel)

        start = time.time()
        R = compute_corner_response(fimg)
        print("compute_corner_reponse time:", time.time() - start)

        corner_raw = R * 255
        corner_raw = corner_raw.astype(np.uint8)
        cv2.imwrite('./result/part_3_corner_raw_' + target, corner_raw)
        cv2.imshow('RAW', corner_raw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        img = img.astype(np.uint8)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        for i in range(0, img_height):
            for j in range(0, img_width):
                if R[i][j] > 0.1:
                    img_color[i][j] = [51, 255, 51]  # green

        cv2.imwrite('./result/part_3_corner_bin_' + target, img_color)
        cv2.imshow('Result', img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        start = time.time()
        suppressed_R = non_maximum_suppression_win(R, 11)
        print("non_maximum_suppression_win time:", time.time() - start)

        suppressed_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        for i in range(0, img_height):
            for j in range(0, img_width):
                if suppressed_R[i][j] > 0:
                    cv2.circle(suppressed_img, (j, i), 5, (51, 255, 51), 2)

        cv2.imwrite('./result/part_3_corner_sup_' + target, suppressed_img)
        cv2.imshow('Result Suppressed', suppressed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
