#hough 변환, 변수 조정을 통한 공검출 예제
import cv2

# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture("0205.png\KakaoTalk_20230223_162409523.mp4")

# Set the frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the parameters for the Hough circle detection
dp = 1 #입력 영상과 축적 배열의 크기 비율. 1이면 동일 크기. 2이면 축적 배열의 가로, 세로 크기가 입력 영상의 반.
minDist = 100000 #검출된 원 중심점들의 최소 거리
param1 = 120 #Canny 에지 검출기의 높은 임계값
param2 = 50  #축적 배열에서 원 검출을 위한 임계값
minRadius = 3  # 검출할 원의 최소, 최대 반지름
maxRadius = 200  # 

# Create a window to display the live camera feed
cv2.namedWindow("Circle Detection")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscaleq
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the grayscale image to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles in the grayscale image using the Hough transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # Draw circles on the original frame
    if circles is not None:
        circles = circles.astype(int)
        for (x, y, r) in circles[0]:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

    # Display the frame in the window
    cv2.imshow("Circle Detection", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
