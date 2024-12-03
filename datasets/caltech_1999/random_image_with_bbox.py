import os
import random
import cv2
import numpy as np


def draw_bbox_on_image(image, bbox):
    points = bbox.reshape((4, 2))  # Convert to 4 points
    for i in range(4):
        start_point = tuple(map(int, points[i]))
        end_point = tuple(map(int, points[(i + 1) % 4]))
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    return image


def create_collage(subdir_data, image_dir):
    # 隨機選擇9張圖片
    folder_names = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]
    selected_images = []

    while len(selected_images) < 9:
        random_folder = random.choice(folder_names)
        folder_path = os.path.join(image_dir, random_folder)
        image_names = [name for name in os.listdir(folder_path) if name.endswith('.jpg')]
        random_image_name = random.choice(image_names)

        # 取得圖片索引
        image_index = int(random_image_name.split('_')[1].split('.')[0]) - 1  # Convert to 0-based index

        # 讀取圖片
        image_path = os.path.join(folder_path, random_image_name)
        image = cv2.imread(image_path)

        if image is not None:
            # 取得邊界框座標
            bbox = subdir_data[:, image_index]
            image_with_bbox = draw_bbox_on_image(image, bbox)
            selected_images.append(image_with_bbox)

    # 拼貼成3x3的圖片
    collage = []
    for i in range(0, 9, 3):
        row = np.hstack(selected_images[i:i+3])
        collage.append(row)
    collage_image = np.vstack(collage)

    # 顯示拼貼圖片
    cv2.imshow("3x3 Image Collage with BBoxes", collage_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用範例
subdir_data = np.load("boundary_boxes.npy")
image_dir = "."  # 圖片目錄
create_collage(subdir_data, image_dir)
