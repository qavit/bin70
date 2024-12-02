"""
This script detects faces and eyes in a video stream using OpenCV's Haar Cascade Classifier.
The script is based on the following tutorial:
https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
"""

import cv2
import argparse
import time
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default=os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt.xml'))
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default=os.path.join(cv2.data.haarcascades, 'haarcascade_eye_tree_eyeglasses.xml'))
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    return parser.parse_args()


def load_cascades(face_cascade_name, eyes_cascade_name):
    face_cascade = cv2.CascadeClassifier()
    eyes_cascade = cv2.CascadeClassifier()

    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)

    return face_cascade, eyes_cascade


def process_video_stream(camera_device, face_cascade, eyes_cascade):
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        detectAndDisplay(frame, face_cascade, eyes_cascade)
        if cv2.waitKey(10) == 27:  # 27 is the ASCII code for the Esc key
            break


def detectAndDisplay(frame, face_cascade, eyes_cascade):
    start_time = time.time()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        faceROI = frame_gray[y:y+h, x:x+w]

        # In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

    end_time = time.time()
    detection_time = end_time - start_time
    print(f"Detection time: {detection_time:.4f} seconds", end="\r")

    cv2.putText(frame, "Press Esc to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Capture - Face detection', frame)


def main():
    args = parse_arguments()
    face_cascade, eyes_cascade = load_cascades(args.face_cascade, args.eyes_cascade)
    print("Press Esc to exit")
    process_video_stream(args.camera, face_cascade, eyes_cascade)


if __name__ == "__main__":
    main()
