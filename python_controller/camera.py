import numpy as np
import cv2
from harvesters.core import Harvester


# Input video frame and output coordinates of detected beads
def detect_beads(image, is_simulator=False):
    h = image.shape[0]
    w = image.shape[1]
    offset = 2

    # Convert image type for opencv compatibility
    image = np.uint8(image)
    if is_simulator:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    step1 = np.zeros((h, w, 1), dtype = "uint8")

    # Preprocessing of image
    cv2.equalizeHist(image, step1)
    cv2.medianBlur(step1, 3, step1)
    cv2.copyMakeBorder(step1, offset, offset, offset, offset, cv2.BORDER_CONSTANT, step1)

    # Initialize blob detector
    params = cv2.SimpleBlobDetector.Params()
    params.minArea = 300
    params.minCircularity = 0.5 #0.85 original value
    params.minInertiaRatio = 0.01
    params.minConvexity = 0.87
    params.filterByArea = 1
    params.filterByInertia = 1
    params.filterByCircularity = 1
    params.filterByColor = 1

    # Detect beads
    blob_detector = cv2.SimpleBlobDetector.create(params)
    key_points = blob_detector.detect(step1)

    return key_points


def camera_main():
    # We can simulate the video feed from the tweezers by playing back a recorded video
    # vid = cv2.VideoCapture(r'.\testing_video1.mp4')
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            # key_points = detect_beads(frame)
            # cv2.drawKeypoints(frame, key_points, frame, (0, 0, 255), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.waitKey(1)
            cv2.imshow('Camera Feed', frame)
        else:
            break


if __name__ == "__main__":
    h = Harvester()
    h.add_file(r'C:\Users\User\Desktop\Tommy_Tweezers_Automation\tweezers_automation\tweezers_automation_v2\bgapi2_gige.cti')
    h.update()
    print(h.device_info_list)

    ia = h.create()
    ia.start()
    while True:
        with ia.fetch() as buffer:
            component = buffer.payload.components[0]
            #print(buffer)
            img = np.ndarray(buffer=component.data.copy(), dtype=np.uint8,
                              shape=(component.height, component.width, 1))
            cv2.waitKey(10)
            cv2.imshow('Camera Feed', img)
