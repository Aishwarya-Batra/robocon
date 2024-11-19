import cv2
import numpy as np

def detect_ball(frame):

    
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for the ball (e.g., orange)
    lower_color = np.array([0, 100, 80])
    upper_color = np.array([10, 180, 160])

    # Create a mask based on the color range
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (which is assumed to be the ball)
    if contours:
        c = max(contours, key=cv2.contourArea)

        # Calculate the center of the ball
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Draw a circle around the ball
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    return frame













# Open the video capture
cap = cv2.VideoCapture("video1.mp4")

currentframe=0
while True:
    ret, frame = cap.read()

    if ret:
        name = './images/frame' + str(currentframe)+'.jpg'
        print('Creating...' + name)
        cv2.imwrite(name, frame)
        currentframe += 1
    else:
        break

    # Detect the ball in the frame
    img = detect_ball(frame)

    # Display the frame
    cv2.imshow("Ball Detection", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()