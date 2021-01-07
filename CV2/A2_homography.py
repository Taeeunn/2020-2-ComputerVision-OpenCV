
import cv2
import numpy as np
import time

# Computing homography with normalization
def compute_homography(srcP, destP):

    N = len(srcP)

    src_sum = srcP.sum(axis=0)
    srcX_mean = src_sum[0] / N
    srcY_mean = src_sum[1] / N

    dest_sum = destP.sum(axis=0)
    destX_mean = dest_sum[0] / N
    destY_mean = dest_sum[1] / N

    # Mean subtraction: translate the mean of the points to the origin (0, 0)
    M1_src = [[1, 0, -srcX_mean], [0, 1, -srcY_mean], [0, 0, 1]]
    M1_dest = [[1, 0, -destX_mean], [0, 1, -destY_mean], [0, 0, 1]]

    for i in range(0, N):
        point = np.dot(M1_src, [[srcP[i][0]], [srcP[i][1]], [1]])
        srcP[i] = [point[0][0], point[1][0]]
        point = np.dot(M1_dest, [[destP[i][0]], [destP[i][1]], [1]])
        destP[i] = [point[0][0], point[1][0]]

    # Scaling: scale the points so that the longest distance to the origin is √2
    sorted_srcP = sorted(srcP, key=lambda x: np.square(x[0]) + np.square(x[1]))
    norm_src = np.sqrt(np.square(sorted_srcP[N - 1][0]) + np.square(sorted_srcP[N - 1][1]))

    sorted_destP = sorted(destP, key=lambda x: np.square(x[0]) + np.square(x[1]))
    norm_dest = np.sqrt(np.square(sorted_destP[N - 1][0]) + np.square(sorted_destP[N - 1][1]))

    M2_src = [[np.sqrt(2) / norm_src, 0, 0], [0, np.sqrt(2) / norm_src, 0], [0, 0, 1]]
    M2_dest = [[np.sqrt(2) / norm_dest, 0, 0], [0, np.sqrt(2) / norm_dest, 0], [0, 0, 1]]

    for i in range(0, N):
        point = np.dot(M2_src, [[srcP[i][0]], [srcP[i][1]], [1]])
        srcP[i] = [point[0][0], point[1][0]]
        point = np.dot(M2_dest, [[destP[i][0]], [destP[i][1]], [1]])
        destP[i] = [point[0][0], point[1][0]]

    Ts = np.dot(M2_src, M1_src)
    Td = np.dot(M2_dest, M1_dest)

    A = []
    for i in range(0, N):
        x = srcP[i][0]
        y = srcP[i][1]
        x_ = destP[i][0]
        y_ = destP[i][1]

        A.append([-x, -y, -1, 0, 0, 0, x * x_, y * x_, x_])
        A.append([0, 0, 0, -x, -y, -1, x * y_, y * y_, y_])

    A = np.array(A)

    U, s, V = np.linalg.svd(A)
    h = np.array(V[len(s) - 1])
    Hn = h.reshape(3, 3)

    H = np.dot(np.dot(np.linalg.inv(Td), Hn), Ts)

    return H


# Computing homography with RANSAC
def compute_homography_ransac(srcP, destP, th):

    N = len(srcP)
    maxInliers = []
    maxnum=0

    start_time = time.time()

    # RANSAC loop to compute the homography matrix
    while True:
        if time.time() - start_time >= 3:
            break

        new_srcP = np.empty((4, 2))
        new_destP = np.empty((4, 2))

        rand_idx = []
        inlier = []

        # randomly select a four point correspondences
        while True:
            rand = np.random.randint(N)
            if rand not in rand_idx:
                rand_idx.append(rand)
            if len(rand_idx)==4:
                break


        for i in range(0, len(rand_idx)):
            new_srcP[i]=srcP[rand_idx[i]]
            new_destP[i]=destP[rand_idx[i]]

        # compute H
        H = compute_homography(new_srcP, new_destP)


        for i in range(N):
            pred = np.dot(H, np.array([srcP[i][0], srcP[i][1], 1]).transpose())
            if pred[2]!=0:
                pred = pred / pred[2]
            dest = np.array([destP[i][0], destP[i][1], 1]).transpose()
            dis = np.linalg.norm(dest - pred)

            if dis < th:
                inlier.append(i)

        if len(inlier) > maxnum:
            maxInliers = np.copy(inlier)
            maxnum = len(inlier)  # count inliers to the current H


    srcP_max = np.empty((maxnum, 2))
    destP_max = np.empty((maxnum, 2))

    for i in range(0, maxnum):
        srcP_max[i]=srcP[maxInliers[i]]
        destP_max[i]=destP[maxInliers[i]]


    H = compute_homography(srcP_max, destP_max)

    return H



def main():
    img_desk = cv2.imread('cv_desk.png', cv2.IMREAD_GRAYSCALE)
    img_cover = cv2.imread('cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
    img_hp = cv2.imread('hp_cover.jpg', cv2.IMREAD_GRAYSCALE)

    #========================================2-1=========================================
    # Feature detection, description, and matching

    orb = cv2.ORB_create()
    kp1 = orb.detect(img_desk, None)
    kp1, des1 = orb.compute(img_desk, kp1)
    kp2 = orb.detect(img_cover, None)
    kp2, des2 = orb.compute(img_cover, kp2)

    matchlist = []  # correspondences
    mindist = 0  # distance
    minqidx = 0  # queryIndex
    mintidx = 0  # trainIndex
    miniidx = 0  # imgIdx

    for i in range(0, len(kp1)):
        for j in range(0, len(kp2)):
            dist = cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING)
            if j == 0:
                mindist = dist
                minqidx = i
                mintidx = j
            else:
                if mindist > dist:
                    mindist = dist
                    minqidx = i
                    mintidx = j

        match = cv2.DMatch(minqidx, mintidx, miniidx, mindist)
        matchlist.append(match)

    matchlist = sorted(matchlist, key=lambda x: x.distance)

    # display top 10 matched pairs according to feature similarities.
    res = cv2.drawMatches(img_desk, kp1, img_cover, kp2, matchlist[:10], None, flags=2)

    cv2.imshow('2-1', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ========================================2-4=========================================
    kp1 = orb.detect(img_cover, None)
    kp1, des1 = orb.compute(img_cover, kp1)
    kp2 = orb.detect(img_desk, None)
    kp2, des2 = orb.compute(img_desk, kp2)

    matchlist = []  # correspondences
    mindist = 0  # distance
    minqidx = 0  # queryIndex
    mintidx = 0  # trainIndex
    miniidx = 0  # imgIdx

    for i in range(0, len(kp1)):
        for j in range(0, len(kp2)):
            dist = cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING)
            if j == 0:
                mindist = dist
                minqidx = i
                mintidx = j
            else:
                if mindist > dist:
                    mindist = dist
                    minqidx = i
                    mintidx = j

        match = cv2.DMatch(minqidx, mintidx, miniidx, mindist)
        matchlist.append(match)

    matchlist = sorted(matchlist, key=lambda x: x.distance)

    srcP = np.float32([kp1[m.queryIdx].pt for m in matchlist])
    destP = np.float32([kp2[m.trainIdx].pt for m in matchlist])

    desk_height, desk_width = img_desk.shape

    now_srcP=np.copy(srcP[0:35])
    now_destP=np.copy(destP[0:35])
    H = compute_homography(now_srcP, now_destP)
    res = cv2.warpPerspective(img_cover, H, (desk_width, desk_height))

    cv2.imshow('Homography with normalization', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    desk_cover = np.copy(img_desk)
    for i in range(desk_height):
        for j in range(desk_width):
            if res[i][j] != 0:  # not black background
                desk_cover[i][j] = res[i][j]

    cv2.imshow('Homography with normalization', desk_cover)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    now_srcP = np.copy(srcP[30:55])
    now_destP = np.copy(destP[30:55])
    H2 = compute_homography_ransac(now_srcP, now_destP, 7)
    res = cv2.warpPerspective(img_cover, H2, (desk_width, desk_height))

    cv2.imshow('Homography with RANSAC', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    desk_cover2 = np.copy(img_desk)

    for i in range(img_desk.shape[0]):
        for j in range(img_desk.shape[1]):
            if res[i][j] != 0:
                desk_cover2[i][j] = res[i][j]

    cv2.imshow('Homography with RANSAC', desk_cover2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    img_hp = cv2.resize(img_hp, (img_cover.shape[1], img_cover.shape[0]))
    res3 = cv2.warpPerspective(img_hp, H2, (desk_width, desk_height))

    cv2.imshow('homography with RANSAC (Harry Potter)', res3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hp_desk = np.copy(img_desk)

    for i in range(desk_height):
        for j in range(desk_width):
            if res3[i][j] != 0:
                hp_desk[i][j] = res3[i][j]

    cv2.imshow('homography with RANSAC (Harry Potter)', hp_desk)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ========================================2-5=========================================
    # Image stitching

    img1 = cv2.imread('diamondhead-10.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('diamondhead-11.png', cv2.IMREAD_GRAYSCALE)

    kp1 = orb.detect(img2, None)
    kp1, des1 = orb.compute(img2, kp1)
    kp2 = orb.detect(img1, None)
    kp2, des2 = orb.compute(img1, kp2)

    matchlist = []  # correspondences
    mindist = 0  # distance
    minqidx = 0  # queryIndex
    mintidx = 0  # trainIndex
    miniidx = 0  # imgIdx

    for i in range(0, len(kp1)):
        for j in range(0, len(kp2)):
            dist = cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING)
            if j == 0:
                mindist = dist
                minqidx = i
                mintidx = j
            else:
                if mindist > dist:
                    mindist = dist
                    minqidx = i
                    mintidx = j

        match = cv2.DMatch(minqidx, mintidx, miniidx, mindist)
        matchlist.append(match)

    matchlist = sorted(matchlist, key=lambda x: x.distance)

    srcP = np.float32([kp1[m.queryIdx].pt for m in matchlist])
    destP = np.float32([kp2[m.trainIdx].pt for m in matchlist])

    now_srcP = np.copy(srcP[30:55])
    now_destP = np.copy(destP[30:55])
    H = compute_homography_ransac(now_srcP, now_destP, 7)

    img1_height, img1_width=img1.shape
    res = cv2.warpPerspective(img2, H, (img1_width + 370, img1_height))
    for i in range(0, img1_height):
        for j in range(0, img1_width):
            res[i][j] = img1[i][j]

    cv2.imshow("Image stitching", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # gradation based blending
    right = cv2.warpPerspective(img2, H, (img1_width + 370, img1_height))
    left = np.zeros((img1_height, img1_width + 370), dtype=float)
    for i in range(0, img1_height):
        const=1
        for j in range(0, img1_width):
            left[i][j] = img1[i][j]
        for j in range(img1_width-100, img1_width):
            const -= 0.01
            res[i][j] = const * left[i][j] + (1 - const) * right[i][j]


    cv2.imshow("Image stitching (with blending)", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
