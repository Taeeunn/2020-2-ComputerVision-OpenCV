import numpy as np

N = 1000 # num of images
D = 1000 # dimensionality of descriptor

def main():
    
    sifts = []  # (1000,n,128)  pre-extracted SIFT features of 1000 images
                # each sift have (n, 128) dim
    centers = np.zeros((D, 128), dtype=np.float32)  # 각 클러스터 (D=1000개) 의 중심 (D,128)
    d = np.zeros((N, D), dtype=np.float32)  # descriptor

    sift_path = "./sift/sift100"
    
    for i in range(0, N):
        f = open(sift_path + str("%03d" % i), 'rb')
        data = f.read()
        sift = np.frombuffer(data, dtype=np.uint8)
        sift = np.reshape(sift, (-1, 128))
        sifts.append(sift) # sift 저장
        centers[i] = np.average(sift, axis=0) # 가장 처음의 center는 각 sift들(1000개)의 평균
        f.close()

    # K-means clustering
    num_epoch=20
    for epoch in range(num_epoch):
        print("epoch", epoch+1)
        new_cluster = []  # 각 SIFT feature 들 (1000 x n개) 이 어느 클러스터로 분류되었는지 저장
        new_cluster_sum = np.zeros((D, 128)) # 각 클러스터로 분류된 SIFT feature들의 합
        new_cluster_cnt = np.zeros((D, 1))  # 각 클러스터로 분류된 SIFT feature들의 개수

        for i in range(0, N):
            #print(i)
            cluster_update= [0] * len(sifts[i])

            for j in range(0, len(sifts[i])):
                dis = np.linalg.norm(sifts[i][j] - centers, axis=1, ord=2)
                min_idx = np.argmin(dis)
                cluster_update[j] = min_idx

                new_cluster_sum[min_idx] += sifts[i][j]
                new_cluster_cnt[min_idx] += 1

            new_cluster.append(cluster_update)


        for i in range(0, N):
            for j in range(0, len(sifts[i])):
                k = new_cluster[i][j]

                # center update
                if new_cluster_cnt[k]==0:
                    centers[k]=0

                else:
                    centers[k] = new_cluster_sum[k] / new_cluster_cnt[k]

                # descriptor update
                d[i, k] -= np.log(np.linalg.norm(sifts[i][j] - centers[k], ord=2) + 1e-20) / len(sifts[i])
                # 거리가 0일 경우를 위해 매우 작은 값을 더해줌


    f = open('./A4_2017313008' + '.des', 'wb')
    N_ = np.array([N], dtype=np.int32)
    D_ = np.array([D], dtype=np.int32)

    f.write(N_.tobytes())
    f.write(D_.tobytes())
    f.write(d.tobytes())
    f.close()


if __name__ == "__main__":
    main()
