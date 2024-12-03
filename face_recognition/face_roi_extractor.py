# Standard library imports
import sys
import os
import pickle
from itertools import groupby

# Third-party imports
import cv2
import numpy as np
from imutils import paths
from tqdm import tqdm

# Local application imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from face_detection.ssd.SSD_webcam import detect_faces


def images_to_faces(input_path):
    """
    將資料集內的照片依序擷取人臉後，轉成灰階圖片，回傳後續可以用作訓練的資料
    :return: (faces, labels)
    """
    # 判斷是否需要重新載入資料
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces.pickle")
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            (faces, labels) = pickle.load(f)
            return (faces, labels)

    # 載入所有圖片
    image_paths = list(paths.list_images(input_path))
    image_paths.sort()

    # 過濾掉層級結構不符合預期的路徑
    valid_image_paths = [path for path in image_paths if len(os.path.normpath(path).split(os.path.sep)) > 1]
    if len(valid_image_paths) == 0:
        raise ValueError(f"No valid image paths found in {input_path}")

    # 將圖片屬於"哪一個人"的名稱取出 (如：man_1, man_2,...)，並以此名稱將圖片分組
    groups = groupby(valid_image_paths, key=lambda path: os.path.normpath(path).split(os.path.sep)[-2])

    # 初始化結果 (faces, labels)
    faces = []
    labels = []

    # 遍歷分好組的圖片
    for name, group_image_paths in groups:
        group_image_paths = list(group_image_paths)

        # 如果樣本小於15，則不考慮使用該人的圖片 (因為會造成辨識結果誤差)；可以嘗試將下面兩行註解看準確度的差異
        if (len(group_image_paths)) < 15:
            continue

        for imagePath in tqdm(group_image_paths, desc=f"[INFO] Processing images for {name}/"):
            # 將圖片依序載入，取得人臉矩形框
            img = cv2.imread(imagePath)
            rects = detect_faces(img)
            # 遍歷各矩形框
            for rect in rects:
                (x, y, w, h) = rect["box"]
                # 取得人臉 ROI (注意在用陣列操作時，順序是 (rows, columns) => 也就是(y, x) )
                roi = img[y:y + h, x:x + w]
                # 將人臉的大小都轉成 50 x 50的圖片
                roi = cv2.resize(roi, (50, 50))
                # 轉成灰階
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # 更新結果
                faces.append(roi)
                labels.append(name)

    # 將結果轉成 NumPy array，方便後續進行訓練
    faces = np.array(faces)
    labels = np.array(labels)

    with open(data_file, "wb") as f:
        pickle.dump((faces, labels), f)

    return (faces, labels)
