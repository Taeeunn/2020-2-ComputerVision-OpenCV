import cv2
import numpy as np

now_point=[]

def get_transformed_image(img,M):
    global now_point
    plane = np.zeros((801, 801)) + 255

    img_width = img.shape[1]
    img_height = img.shape[0]

    print("------------------------------------------------------------")
    print(M)

    # If M is identity matrix -> move to origin
    if np.array_equal(M, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        origin_point = []
        for i in range(0,img_height):
            for j in range(0,img_width):
                if img[i][j] < 255:   # 흰 색 배경이 아닐 때
                    origin_point.append([j-int(img_width/2),int(img_height/2)-i,1])  # smile 이미지의 중심 = 원점(0, 0)
        now_point = np.copy(origin_point)


    pre_point = now_point
    now_point = np.dot(pre_point,M)

    for point in now_point:
        # (0, 0) is corresponding to the pixel at(400, 400)
        plane[400-int(point[1])][400+int(point[0])]=0

    # draw two arrows to visualize 𝑥𝑥 and 𝑦𝑦 axes.
    cv2.arrowedLine(plane, (0, 400), (800, 400), color=0, thickness=2, tipLength=0.02)
    cv2.arrowedLine(plane, (400, 800), (400, 0), color=0, thickness=2, tipLength=0.02)

    return plane


def main():

    img = cv2.imread('smile.png', cv2.IMREAD_GRAYSCALE)

    M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    plane = get_transformed_image(img, M)
    cv2.imshow('res', plane)
    key = cv2.waitKey(0)

    while True:
        if key == ord('a'):   # Move to the left by 5 pixels
            M=[[1, 0, 0], [0, 1, 0], [-5, 0, 1]]
        elif key == ord('d'):  # Move to the right by 5 pixels
            M=[[1, 0, 0], [0, 1, 0], [5, 0, 1]]
        elif key == ord('w'):  # Move to the upward by 5 pixels
            M=[[1, 0, 0], [0, 1, 0], [0, 5, 1]]
        elif key == ord('s'):  # Move to the downward by 5 pixels
            M = [[1, 0, 0], [0, 1, 0], [0, -5, 1]]
        elif key == ord('r'):  # Rotate counter-clockwise by 5 degrees
            rad = np.radians(5)
            cos = np.cos(rad)
            sin = np.sin(rad)
            M = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        elif key == ord('R'):    # Rotate clockwise by 5 degrees
            rad = np.radians(-5)
            cos = np.cos(rad)
            sin = np.sin(rad)
            M = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        elif key == ord('f'):  # Flip across y axis
            M=[[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        elif key == ord('F'):  # Flip across x axis
            M=[[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        elif key == ord('x'):  # Shrink the size by 5% along to x direction
            M = [[0.95, 0, 0], [0, 1, 0], [0, 0, 1]]
        elif key == ord('X'):  # Enlarge the size by 5% along to x direction
            M = [[1.05, 0, 0], [0, 1, 0], [0, 0, 1]]
        elif key == ord('y'):  # Shrink the size by 5% along to y direction
            M = [[1, 0, 0], [0, 0.95, 0], [0, 0, 1]]
        elif key == ord('Y'):  # Enlarge the size by 5% along to y direction
            M = [[1, 0, 0], [0, 1.05, 0], [0, 0, 1]]
        elif key == ord('H'):  # Restore to the initial state
            M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif key == ord('Q'):  # Quit
            cv2.destroyAllWindows()
            break

        if key == ord('H'):
            plane = get_transformed_image(img, M)
        else:
            plane = get_transformed_image(plane, M)

        cv2.imshow('res', plane)
        key = cv2.waitKey(0)



if __name__ == "__main__":
    main()

