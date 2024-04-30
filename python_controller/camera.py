import numpy as np
import cv2


# Input video frame and output coordinates of detected beads
def detect_beads(image):
    h = image.shape[0]
    w = image.shape[1]
    offset = 2

    # Convert image type for opencv compatibility
    image = np.uint8(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    step1 = np.zeros((h, w, 1), dtype = "uint8")

    # Preprocessing of image
    cv2.equalizeHist(image, step1)
    cv2.medianBlur(step1, 3, step1)
    cv2.copyMakeBorder(step1, offset, offset, offset, offset, cv2.BORDER_CONSTANT, step1)

    # Initialize blob detector
    params = cv2.SimpleBlobDetector.Params()
    params.minArea = 300
    params.minCircularity = 0.85
    params.filterByArea = 1
    params.filterByInertia = 1
    params.filterByCircularity = 1
    params.filterByColor = 1

    # Detect beads
    blob_detector = cv2.SimpleBlobDetector.create(params)
    key_points = blob_detector.detect(step1)

    return key_points


if __name__ == "__main__":
    # We can simulate the video feed from the tweezers by playing back a recorded video
    vid = cv2.VideoCapture(r'.\testing_video1.mp4')

    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            key_points = detect_beads(frame)
            cv2.drawKeypoints(frame, key_points, frame, (0, 0, 255), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.waitKey(1)
            cv2.imshow('Camera Feed', frame)
        else:
            break

