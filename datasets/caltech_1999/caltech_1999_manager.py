"""
This script downloads, extracts, and processes the Caltech Face Dataset 1999.

Example usage:
python caltech_1999_manager.py
python caltech_1999_manager.py --collage
"""


import os
import random
import cv2
import numpy as np
from scipy.io import loadmat
import shutil
import tarfile
import urllib.request
import json
import argparse

DATASET_URL = "https://data.caltech.edu/records/6rjah-hdv18/files/faces.tar?download=1"
TAR_FILE = "faces.tar"
METADATA = "ImageData.mat"
BOUNDING_BOXES = "bbox_data.npy"
RULES = "rules.json"


# Download and extract dataset if not present
def download_and_extract_dataset(url, tar_path, extract_path):
    if not os.path.exists(METADATA):
        print("Dataset not found. Downloading...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete. Extracting...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_path, filter=tarfile.TarInfo)
        print("Extraction complete.")
        os.remove(tar_path)  # Delete the tar file after extraction
    else:
        print("Dataset already exists.")


# Load metadata (ImageData.mat) and save boudning boxes data as .npy
def load_metadata_and_save_bbox(mat_file_path, npy_file_path="bbox_data.npy", bbox_key="SubDir_Data"):
    # Load the .mat file
    data = loadmat(mat_file_path)
    # Extract the SubDir_Data matrix
    subdir_data = data[bbox_key]
    # Save the matrix as a .npy file
    np.save(npy_file_path, subdir_data)
    return subdir_data


# Move images into folders based on rules
def move_images(source_dir, target_dir, rules):
    # Check if files are already moved
    if all(os.path.exists(os.path.join(target_dir, folder_name)) for folder_name in rules.keys()):
        print("All files have already been moved according to rules.json.")
        return

    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    # Save log
    folder_log = []

    # Start moving files
    for folder_name, (start, end) in rules.items():
        # Create folder
        folder_path = os.path.join(target_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Move files
        for i in range(start, end + 1):
            filename = f"image_{i:04d}.jpg"
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(folder_path, filename)

            if os.path.exists(source_path):
                shutil.move(source_path, target_path)

        # Log folder info
        folder_log.append({
            "folder": folder_name,
            "images_count": end - start + 1,
            "range": f"image_{start:04d}.jpg ~ image_{end:04d}.jpg"
        })

    # Print log
    for log in folder_log:
        print(f"Folder: {log['folder']}, Image count: {log['images_count']}, Range: {log['range']}")

    # Save log to file
    log_file = os.path.join(target_dir, "folder_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        for log in folder_log:
            f.write(f"Folder: {log['folder']}, Image count: {log['images_count']}, Range: {log['range']}\n")


# Randomly select 9 images, draw bounding boxes, and create a collage
def draw_bbox_on_image(image, bbox, filename):
    points = bbox.reshape((4, 2))  # Convert to 4 points
    for i in range(4):
        start_point = tuple(map(int, points[i]))
        end_point = tuple(map(int, points[(i + 1) % 4]))
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    # Draw filename with black background
    cv2.rectangle(image, (5, 5), (150, 35), (0, 0, 0), -1)
    cv2.putText(image, filename, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def remove_specified_images(start, end, directory="."):
    for i in range(start, end + 1):
        filename = f"image_{i:04d}.jpg"
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed {filename}")
        else:
            print(f"File {filename} not found. It might have been already removed.")


def create_collage(subdir_data, image_dir):
    # Randomly select 9 images
    folder_names = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]
    selected_images = []

    while len(selected_images) < 9:
        random_folder = random.choice(folder_names)
        folder_path = os.path.join(image_dir, random_folder)
        image_names = [name for name in os.listdir(folder_path) if name.endswith('.jpg')]
        random_image_name = random.choice(image_names)

        # Get image index
        image_index = int(random_image_name.split('_')[1].split('.')[0]) - 1  # Convert to 0-based index

        # Read image
        image_path = os.path.join(folder_path, random_image_name)
        image = cv2.imread(image_path)

        if image is not None:
            # Get bounding box coordinates
            bbox = subdir_data[:, image_index]
            image_with_bbox = draw_bbox_on_image(image, bbox, random_image_name)
            selected_images.append(image_with_bbox)

    # Create 3x3 collage
    collage = []
    for i in range(0, 9, 3):
        row = np.hstack(selected_images[i:i+3])
        collage.append(row)
    collage_image = np.vstack(collage)

    # Display collage
    cv2.imshow("3x3 Image Collage with BBoxes", collage_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save collage image
    cv2.imwrite("collage_with_bboxes.jpg", collage_image)


# Main execution
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Oraganize the Caltech Face Dataset 1999and visualize face images with bounding boxes.")
    parser.add_argument("--collage", "-c", action="store_true", help="Create a collage of 9 random images with bounding boxes.")
    args = parser.parse_args()

    # Download and extract dataset if necessary
    download_and_extract_dataset(DATASET_URL, TAR_FILE, ".")

    # Load and save .mat file
    subdir_data = load_metadata_and_save_bbox(METADATA, BOUNDING_BOXES)

    # Define source and target directories
    source_dir = "."  # Source directory for images
    target_dir = "."  # Target directory for organized images

    # Load rules from JSON file
    with open(RULES, "r") as f:
        rules = json.load(f)

    # Move images according to rules
    move_images(source_dir, target_dir, rules)

    # Remove specified images
    remove_specified_images(399, 403, source_dir)

    # Create and display collage if requested
    if args.collage:
        create_collage(subdir_data, target_dir)
