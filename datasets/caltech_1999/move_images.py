import os
import shutil

# 定義目錄和對應範圍
rules = {
    "man_1": (1, 21),
    "man_2": (22, 41),
    "man_3": (42, 46),
    "man_4": (47, 68),
    "woman_1": (69, 89),
    "man_5": (90, 112),
    "woman_2": (113, 132),
    "man_6": (133, 137),
    "man_7": (138, 158),
    "man_8": (159, 165),
    "woman_3": (166, 170),
    "woman_4": (171, 175),
    "woman_5": (176, 195),
    "man_9": (196, 216),
    "man_10": (217, 241),
    "man_11": (242, 263),
    "man_12": (264, 268),
    "woman_6": (269, 287),
    "man_13": (288, 307),
    "man_14": (308, 336),
    "woman_7": (337, 356),
    "woman_8": (357, 376),
    "man_15": (377, 398),
    "man_16": (404, 408),
    "woman_9": (409, 428),
    "woman_10": (429, 450),
}

# 設定來源目錄和目標目錄
source_dir = "."  
target_dir = "." 

# 確保目標目錄存在
os.makedirs(target_dir, exist_ok=True)

# 儲存記錄
folder_log = []

# 開始移動檔案
for folder_name, (start, end) in rules.items():
    # 建立資料夾
    folder_path = os.path.join(target_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # 移動檔案
    for i in range(start, end + 1):
        filename = f"image_{i:04d}.jpg"
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(folder_path, filename)
        
        if os.path.exists(source_path):
            shutil.move(source_path, target_path)
    
    # 記錄資料夾資訊
    folder_log.append({
        "folder": folder_name,
        "images_count": end - start + 1,
        "range": f"image_{start:04d}.jpg ~ image_{end:04d}.jpg"
    })

# 印出記錄
for log in folder_log:
    print(f"資料夾: {log['folder']}, 照片數量: {log['images_count']}, 範圍: {log['range']}")

# 儲存記錄到檔案
log_file = os.path.join(target_dir, "folder_log.txt")
with open(log_file, "w", encoding="utf-8") as f:
    for log in folder_log:
        f.write(f"資料夾: {log['folder']}, 照片數量: {log['images_count']}, 範圍: {log['range']}\n")
