#이미지 roi 지정시에 좌표읽는 코드
import cv2

img = cv2.imread('0205.png\\ballvideocap2.jpg')

x_pos, y_pos, width, height = cv2.selectROI("location", img, False)
print("x position, y position : ", x_pos, y_pos)
print("width, height :", width, height)

cv2.destroyAllWindows()