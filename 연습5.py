import cv2

# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture('0205.png\KakaoTalk_20230223_162409523.mp4')

# Set the frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the parameters for the Hough circle detection
dp = 1
minDist = 500000000000000000
param1 = 100
param2 = 49  # increase this value
minRadius = 3  # increase this value
maxRadius = 190  # increase this value

# Create a window to display the live camera feed
cv2.namedWindow("Circle Detection")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
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
