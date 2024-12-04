"""
LBP (Local Binary Patterns) 臉部辨識
cv2.face_LBPHFaceRecognizer() 是 OpenCV 中用於臉部辨識的 LBPH 演算法。
"""


# Standard library imports
import sys
import os
import argparse
import random
import time
import glob

# Third-party imports
import cv2
import numpy as np
import imutils
from skimage import feature
from skimage.exposure import rescale_intensity
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Local application imports
from face_roi_extractor import images_to_faces
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from face_detection.ssd.SSD_webcam import detect_faces


def show_lbp_result(input_path, num_random_images=1):
    for _ in range(num_random_images):
        # 隨機選取一張照片來看LBP的結果
        image_paths = glob.glob(os.path.join(input_path, "*", "*.jpg"))
        if not image_paths:
            print(f"[ERROR] No image files found in directory: {input_path}")
            return
        image_path = random.choice(image_paths)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Failed to load image: {image_path}")
            return

        rects = detect_faces(image)

        if not rects:
            print(f"[ERROR] No faces detected in image: {image_path}")
            return

        (x, y, w, h) = rects[0]["box"]
        roi = image[y:y + h, x:x + w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, 8, 1, method="default")
        lbp = rescale_intensity(lbp, out_range=(0, 255))
        lbp = lbp.astype("uint8")

        path_parts = os.path.normpath(image_path).split(os.sep)
        short_path = os.path.join(path_parts[-2], path_parts[-1])

        print(f"[INFO] Displaying LBP image of {short_path}")
        img_with_lbp = np.hstack([roi, np.dstack([lbp] * 3)])
        cv2.imshow(f"{short_path} and its LBP", img_with_lbp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def train_and_predict(input_path, num_random_images=1):
    print("[INFO] Loading pickle dataset...", end="")
    (faces, labels) = images_to_faces(input_path)
    print(f" Found {len(faces)} images")

    # 將名稱從字串轉成整數 (在做訓練時時常會用到這個方法：label encoding)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # 將資料拆分訓練用與測試用；測試資料佔總資料1/4 (方便後續我們判斷這個方法的準確率)
    split = train_test_split(faces, labels, test_size=0.25, stratify=labels, random_state=9527)
    (trainX, testX, trainY, testY) = split

    print("[INFO] Training...", end="")
    start = time.time()
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(trainX, trainY)
    end = time.time()
    print(f" took {round(end - start, 2)} seconds")

    # 辨識測試資料
    print("[INFO] Predicting...", end="")
    start = time.time()
    predictions = []
    confidence = []

    # loop over the test data
    for i in range(0, len(testX)):
        (prediction, conf) = recognizer.predict(testX[i])
        predictions.append(prediction)
        confidence.append(conf)
    end = time.time()
    print(f" took: {round(end - start, 2)} seconds")

    print("[INFO] Classification report:")
    print(classification_report(testY, predictions, target_names=le.classes_))

    # 隨機挑選測試資料來看結果
    idxs = np.random.choice(range(0, len(testY)), size=num_random_images, replace=False)
    for i in idxs:
        predName = le.inverse_transform([predictions[i]])[0]
        actualName = le.classes_[testY[i]]

        face = np.dstack([testX[i]] * 3)
        face = imutils.resize(face, width=250)

        cv2.putText(face, f"pred: {predName}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(face, f"actual:{actualName}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        print(f"[INFO] Predicted: {predName}, Actual: {actualName}")
        cv2.imshow("Face", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True, help="the input dataset path")
    ap.add_argument("-n", "--num_images", type=int, default=3, help="the number of random images to show")
    args = vars(ap.parse_args())

    show_lbp_result(args["input"], args["num_images"])
    train_and_predict(args["input"], args["num_images"])
