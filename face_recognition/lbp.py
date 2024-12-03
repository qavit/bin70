"""
LBP (Local Binary Patterns) 臉部辨識
cv2.face_LBPHFaceRecognizer() 是 OpenCV 中用於臉部辨識的 LBPH 演算法。
"""


# Standard library imports
import sys
import os
import argparse
# import random
import time

# Third-party imports
import cv2
import numpy as np
import imutils
# from skimage import feature
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add the parent directory to the Python path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local application imports
from datasets.face_roi_extractor import images_to_faces


def main():
    # 初始化arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True, help="the input dataset path")
    args = vars(ap.parse_args())

    print("[INFO] Loading dataset....")
    (faces, labels) = images_to_faces(args["input"])
    print(f"[INFO] Found {len(faces)} images in dataset")

    # 將名稱從字串轉成整數 (在做訓練時時常會用到這個方法：label encoding)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # 將資料拆分訓練用與測試用；測試資料佔總資料1/4 (方便後續我們判斷這個方法的準確率)
    split = train_test_split(faces, labels, test_size=0.25, stratify=labels, random_state=9527)
    (trainX, testX, trainY, testY) = split

    print("[INFO] Training...")
    start = time.time()
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(trainX, trainY)
    end = time.time()
    print(f"[INFO] Training took {round(end - start, 2)} seconds")

    # 辨識測試資料
    print("[INFO] Predicting...")
    start = time.time()
    predictions = []
    confidence = []

    # loop over the test data
    for i in range(0, len(testX)):
        (prediction, conf) = recognizer.predict(testX[i])
        predictions.append(prediction)
        confidence.append(conf)
    end = time.time()
    print(f"[INFO] Training took: {round(end - start, 2)} seconds")
    print(classification_report(testY, predictions, target_names=le.classes_))

    # 隨機挑選測試資料來看結果
    idxs = np.random.choice(range(0, len(testY)), size=10, replace=False)
    for i in idxs:
        predName = le.inverse_transform([predictions[i]])[0]
        actualName = le.classes_[testY[i]]

        face = np.dstack([testX[i]] * 3)
        face = imutils.resize(face, width=250)

        cv2.putText(face, f"pred:{predName}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(face, f"actual:{actualName}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        print(f"[INFO] predicted: {predName}, actual: {actualName}")
        cv2.imshow("Face", face)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
