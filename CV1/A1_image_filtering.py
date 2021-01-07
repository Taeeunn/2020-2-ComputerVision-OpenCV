import cv2
import numpy as np
import my_functions as my
import time


def main():

    print("----------------------------------------------------------------------\n")
    print("1D Gaussian Kernel (5, 1)")
    print(my.get_gaussian_filter_1d(5, 1))
    print("\n2D Gaussian Kernel (5, 1)")
    print(my.get_gaussian_filter_2d(5, 1))

    for target in ['shapes.png', 'lenna.png']:
        print("\n----------------------------------------------------------------------")
        print("<", target, ">\n")
        img = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
        cv2.imshow(target, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, 40)
        fontscale = 1
        color = (0, 0, 0)
        thickness = 2

        for i in range(5, 18, 6):
            for j in range(1, 12, 5):
                print(str(i) + "x" + str(i) + " s=" + str(j))
                kernel = my.get_gaussian_filter_2d(i, j)
                np_fimg = my.cross_correlation_2d(img, kernel)
                fimg = np_fimg.astype(np.uint8)
                text = str(i) + "x" + str(i) + " s=" + str(j)
                cv2.putText(fimg, text, org, font, fontscale, color, thickness, cv2.LINE_AA)
                if j == 1:
                    row_img = fimg
                else:
                    row_img = cv2.hconcat([row_img, fimg])

            if i == 5:
                result_img = row_img
            else:
                result_img = cv2.vconcat([result_img, row_img])

        cv2.imwrite('./result/part_1_gaussian_filtered_' + target, result_img)
        cv2.imshow('Gaussian Filtering Result', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 1D correlation kernel (Gaussian Filter)
        kernel_vertical = my.get_gaussian_filter_1d(5, 11).reshape(-1, 1)
        kernel_horizontal = my.get_gaussian_filter_1d(5, 11)

        start1 = time.time()
        fimg_1d = my.cross_correlation_1d(img, kernel_vertical)  # cross-correlation between an image and a 1D kernel (vertical):
        time_vertical = time.time() - start1

        start2 = time.time()
        fimg_1d = my.cross_correlation_1d(fimg_1d, kernel_horizontal)  # cross-correlation between an image and a 1D kernel (horizontal):
        time_horizontal = time.time() - start2

        print("\n1d filtering(vertical) time:", time_vertical)
        print("1d filtering(horizontal) time:", time_horizontal)
        print("1d filtering(vertical->horizontal) time:", time_vertical + time_horizontal)

        kernel_2d = my.get_gaussian_filter_2d(5, 11)
        start = time.time()
        fimg_2d = my.cross_correlation_2d(img, kernel_2d)
        time_2d = time.time() - start

        print("\n2d filtering time:", time_2d)

        difference = cv2.absdiff(fimg_2d, fimg_1d)  # pixel-wise difference map
        diff_sum = np.sum(difference)
        difference = difference.astype(np.uint8)
        cv2.imshow('pixel-wise difference map', difference)
        print("\nsum of (absolute) intensity differences:", diff_sum)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
