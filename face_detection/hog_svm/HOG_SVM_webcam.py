"""
Purpose: Use Dlib's HOG + Linear SVM model to detect faces in real-time from WebCam images.

Credits:
Rewritten from iT 邦幫忙 tutorial article [Day11] Face Detection - Using OpenCV & Dlib: Dlib HOG + Linear SVM
URL: https://ithelp.ithome.com.tw/articles/10264199
Author: Uncle Sam
Original code: https://github.com/saivirtue/face_under_computer_vision/blob/main/face_detection/dlib_hog_svm.py

Usage:
- The program will automatically start the WebCam and begin real-time detection.
- Press the "q" key to exit the program.
"""

# Import necessary packages
import time

import cv2
import dlib
import imutils
import numpy as np
from imutils.face_utils import rect_to_bb
from imutils.video import WebcamVideoStream


# Initialize the model
detector = dlib.get_frontal_face_detector()


# Define a face detection function for reuse
def detect(img, return_ori_result=False):
    # Detect faces and convert the results to (x, y, w, h) bounding boxes
    results = detector(img, 0)
    rects = [rect_to_bb(rect) for rect in results]
    return results if return_ori_result else rects


def main():
    # Start the WebCam
    vs = WebcamVideoStream().start()
    time.sleep(2.0)
    start = time.time()
    fps = vs.stream.get(cv2.CAP_PROP_FPS)
    print("Frames per second using cv2.CAP_PROP_FPS : {0}".format(fps))

    while True:
        # Get the current frame, resize it to a width of 300, and convert it to an RGB image
        frame = vs.read()
        img = frame.copy()
        img = imutils.resize(img, width=300)

        # Get the size of the frame (height, width)
        ratio = frame.shape[1] / img.shape[1]

        # Call the detection function to get results
        rects = detect(img)

        # All prediction results
        for rect in rects:
            # Calculate the bounding box and accuracy - get the values (top-left X, top-left Y, bottom-right X, bottom-right Y) (remember to convert back to the original frame size)
            box = np.array(rect) * ratio
            (x, y, w, h) = box.astype("int")

            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display FPS
        end = time.time()
        cv2.putText(frame, f"FPS: {str(int(1 / (end - start)))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Press Q to exit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        start = end

        # Show the image
        cv2.imshow("Frame", frame)

        # Check if "q" is pressed; exit the loop
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
