import cv2
import numpy as np
import compute_avg_reproj_error as error
import time

global img1_height, img1_width, img2_height, img2_width

img = [['temple1.png', 'temple2.png'], ['house1.jpg', 'house2.jpg'], ['library1.jpg', 'library2.jpg']]
match = ['temple_matches.txt', 'house_matches.txt', 'library_matches.txt']


# Eight-point algorithm to compute the fundamental matrix
def compute_F_raw(M):
    A = []
    for row in M:
        x1, y1, x2, y2 = row
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])

    U, s, V = np.linalg.svd(A)

    F = np.reshape(V[len(s) - 1], (3, -1))

    return F

# Eight-point algorithm with a normalization
def compute_F_norm(M):
    global img1_height, img1_width, img2_height, img2_width

    N = len(M)

    img1_height_half = img1_height / 2
    img1_width_half = img1_width / 2
    img2_height_half = img2_height / 2
    img2_width_half = img2_width / 2

    # normalization (independent from the feature locations is recommended)

    # 1. Translation to move the image center to origin(0, 0)
    trans_1 = [[1, 0, -img1_width_half], [0, 1, -img1_height_half], [0, 0, 1]]
    trans_2 = [[1, 0, -img2_width_half], [0, 1, -img2_height_half], [0, 0, 1]]

    # 2. Scaling to fit the image into an unit square [(-1, -1), [+1, +1)]
    scale_1 = [[1 / img1_width_half, 0, 0], [0, 1 / img1_height_half, 0], [0, 0, 1]]
    scale_2 = [[1 / img2_width_half, 0, 0], [0, 1 / img2_height_half, 0], [0, 0, 1]]

    T_1 = np.dot(scale_1, trans_1)
    T_2 = np.dot(scale_2, trans_2)

    M_norm = np.empty(M.shape)

    for i in range(0, N):
        point1 = np.dot(T_1, [[M[i][0]], [M[i][1]], [1]])
        M_norm[i][0], M_norm[i][1] = point1[0][0], point1[1][0]
        point2 = np.dot(T_2, [[M[i][2]], [M[i][3]], [1]])
        M_norm[i][2], M_norm[i][3] = point2[0][0], point2[1][0]

    A = []
    for row in M_norm:
        x1, y1, x2, y2 = row
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])

    U, s, V = np.linalg.svd(A)

    F = np.reshape(V[len(s) - 1], (3, -1))

    # U, s, V = np.linalg.svd(F)
    # em=np.zeros((3, 3))
    # em[0][0]=s[0]
    # em[1][1]=s[1]
    # print(em)
    #
    # F = np.dot(np.dot(U, em), V)

    F = np.dot(np.dot(np.transpose(T_2), F), T_1)

    return F


def compute_F_mine(M):
    global img1_height, img1_width, img2_height, img2_width

    start_time = time.time()
    N = len(M)

    img1_height_half = img1_height / 2
    img1_width_half = img1_width / 2
    img2_height_half = img2_height / 2
    img2_width_half = img2_width / 2

    # normalization (independent from the feature locations is recommended)

    # 1. Translation to move the image center to origin(0, 0)
    trans_1 = [[1, 0, -img1_width_half], [0, 1, -img1_height_half], [0, 0, 1]]
    trans_2 = [[1, 0, -img2_width_half], [0, 1, -img2_height_half], [0, 0, 1]]

    # 2. Scaling to fit the image into an unit square [(-1, -1), [+1, +1)]
    scale_1 = [[1 / img1_width_half, 0, 0], [0, 1 / img1_height_half, 0], [0, 0, 1]]
    scale_2 = [[1 / img2_width_half, 0, 0], [0, 1 / img2_height_half, 0], [0, 0, 1]]

    T_1 = np.dot(scale_1, trans_1)
    T_2 = np.dot(scale_2, trans_2)

    M_norm = np.empty(M.shape)

    for i in range(0, N):
        point1 = np.dot(T_1, [[M[i][0]], [M[i][1]], [1]])
        M_norm[i][0], M_norm[i][1] = point1[0][0], point1[1][0]
        point2 = np.dot(T_2, [[M[i][2]], [M[i][3]], [1]])
        M_norm[i][2], M_norm[i][3] = point2[0][0], point2[1][0]

    min_error = np.inf
    F_final = None

    while True:
        if time.time() - start_time >= 3:  # should return the result within 3 seconds.
            break

        A = []
        cor_idx = np.random.choice(len(M), 9, replace=False)
        # cor_idx = np.arange(M.shape[0])
        # np.random.shuffle(cor_idx)
        # cor_idx = cor_idx[:9]

        for i in range(0, 9):
            idx = cor_idx[i]
            x1, y1, x2, y2 = M_norm[idx]
            A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])

        U, s, V = np.linalg.svd(A)

        F = np.reshape(V[len(s) - 1], (3, -1))

        F = np.dot(np.dot(np.transpose(T_2), F), T_1)

        err = error.compute_avg_reproj_error(M, F)

        if min_error > err:
            min_error = err
            F_final = F

    return F_final


def main():
    global img1_height, img1_width, img2_height, img2_width

    # 1-1. Fundamental matrix computation
    print('[1-1. Fundamental matrix computation]')

    for i in range(0, len(img)):
        print('-----------------------------------------------------------')
        print('<' + img[i][0] + ' and ' + img[i][1] + '>')

        img1 = cv2.imread(img[i][0], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img[i][1], cv2.IMREAD_GRAYSCALE)

        img1_height = img1.shape[0]
        img1_width = img1.shape[1]
        img2_height = img2.shape[0]
        img2_width = img2.shape[1]

        matches = np.loadtxt(match[i])

        F_raw = compute_F_raw(matches)
        F_norm = compute_F_norm(matches)
        F_mine = compute_F_mine(matches)

        # Print the average reprojection errors
        print('Average Reprojection Errors (' + img[i][0] + ' and ' + img[i][1] + ')')
        print("Raw =", error.compute_avg_reproj_error(matches, F_raw))
        print("Norm =", error.compute_avg_reproj_error(matches, F_norm))
        print("Mine =", error.compute_avg_reproj_error(matches, F_mine))



    # 1-2. Visualization of epipolar lines
    print('\n[1-2. Visualization of epipolar lines]')

    for i in range(0, len(img)):

        print('-----------------------------------------------------------')
        print('<' + img[i][0] + ' and ' + img[i][1] + '>')

        # Load two images
        img1 = cv2.imread(img[i][0], cv2.IMREAD_COLOR)
        img2 = cv2.imread(img[i][1], cv2.IMREAD_COLOR)

        img1_org = img1.copy()
        img2_org = img2.copy()

        img1_height = img1.shape[0]
        img1_width = img1.shape[1]
        img2_height = img2.shape[0]
        img2_width = img2.shape[1]

        # feature correspondences between two images
        matches = np.loadtxt(match[i])

        F = compute_F_mine(matches)

        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)

        color = [red, green, blue]

        while True:
            # Randomly select 3 correspondances
            cor_idx = np.random.choice(len(matches), 3, replace=False)

            for j in range(0, 3):
                idx = cor_idx[j]

                x1, y1 = matches[idx][0], matches[idx][1]  # p1
                x2, y2 = matches[idx][2], matches[idx][3]  # q1

                l = np.dot(F, [[x1], [y1], [1]])  # epipolar line in the second view
                m = np.dot(np.transpose(F), [[x2], [y2], [1]])  # epipolar line in the first view

                # 2차원 공간에서 line: ax + by + c = 0
                # y = -(a/b)x - (c/b)
                a1, b1, c1 = m  # first view
                a2, b2, c2 = l  # second view

                cv2.circle(img1, (int(x1), int(y1)), 3, color[j], 3)
                cv2.circle(img2, (int(x2), int(y2)), 3, color[j], 3)
                cv2.line(img1, (0, -int(c1 / b1)), (img1_width, int(-(a1 / b1) * img1_width - (c1 / b1))), color[j], 2)
                cv2.line(img2, (0, -int(c2 / b2)), (img2_width, int(-(a2 / b2) * img2_width - (c2 / b2))), color[j], 2)

            result_img = cv2.hconcat([img1, img2])
            cv2.imshow("1-2", result_img)
            key = cv2.waitKey(0)

            if key == ord('q'):
                cv2.destroyAllWindows()
                break

            img1 = img1_org.copy()
            img2 = img2_org.copy()


if __name__ == "__main__":
    main()
