"""
This script is used to detect faces in an image using MTCNN.
"""

import argparse
from mtcnn import MTCNN
import cv2
import os


def detect_faces_in_image(image_path):
    # Initialize MTCNN
    detector = MTCNN()

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = detector.detect_faces(image_rgb)

    # Draw detection results
    for result in results:
        x, y, width, height = result['box']
        keypoints = result['keypoints']

        # Draw face bounding box
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Draw keypoints
        for key, point in keypoints.items():
            cv2.circle(image, point, 5, (0, 0, 255), -1)

    # Display the result
    cv2.imshow("MTCNN Detection", image)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    default_image_path = os.path.join("..", "assets", "meme_faces", "smart_guy.png")

    parser = argparse.ArgumentParser(description="Detect faces in an image using MTCNN.")
    parser.add_argument("image_path", type=str, nargs='?', default=default_image_path, help="Path to the image file.")
    args = parser.parse_args()

    detect_faces_in_image(args.image_path)
