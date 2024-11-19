import cv2
import numpy as np
import os




def readvideo(videopath, selected_dir, notselected_dir):
    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(videopath))[0]

    # Create subdirectories for this video in the selected and notselected directories
    video_selected_dir = os.path.join(selected_dir, video_name)
    video_notselected_dir = os.path.join(notselected_dir, video_name)

    os.makedirs(video_selected_dir, exist_ok=True)
    os.makedirs(video_notselected_dir, exist_ok=True)

    cam = cv2.VideoCapture(videopath)
    if not cam.isOpened():
        print(f"Error: Could not open video file {videopath}.")
        return

    currentframe = 0
    count1 = 0
    count2 = 0

    # Frame-by-frame processing
    while True:
        ret, frame = cam.read()
        if not ret:
            print(f"End of video: {videopath}")
            break

        # Frame saving
        frame_name = f'frame{currentframe}.jpg'
        print(f"Processing... {frame_name}")
        currentframe += 1

        # Convert to HSV and apply a mask for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([0, 100, 80])
        upper_color = np.array([10, 180, 160])
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Blur the mask for better circle detection
        blurred = cv2.blur(mask, (5, 5), 0)

        # Detect circles in the frame
        detected_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=20, minRadius=35, maxRadius=80
        )

        # Save frames to appropriate directories
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                # Draw the circle and center on the frame
                cv2.circle(frame, (a, b), r, (0, 255, 0), 2)
                cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)
            # Save to the video-specific 'selected' subdirectory
            cv2.imwrite(os.path.join(video_selected_dir, f"image{count1}.jpg"), frame)
            count1 += 1
        else:
            # Save to the video-specific 'notselected' subdirectory
            cv2.imwrite(os.path.join(video_notselected_dir, f"image{count2}.jpg"), frame)
            count2 += 1

        # Optional: Display the frame (can be removed for batch processing)
        cv2.imshow("Detected Circles", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close any windows
    cam.release()
    cv2.destroyAllWindows()

video1 = 'video1.mp4'
video2 = 'video2.mp4'

readvideo(video1, r'D:\robocon\selected', r'D:\robocon\notselected')
readvideo(video2, r'D:\robocon\selected', r'D:\robocon\notselected')

