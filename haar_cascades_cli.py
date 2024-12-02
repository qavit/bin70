import cv2
import argparse


def detect_faces_in_image(image_path, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    # 1. Load the Haar feature classifier XML file
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    # 2. Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")

    # 3. Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. Perform face detection
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )

    print(f"Found {len(faces)} faces")

    # 5. Draw green rectangles around detected faces
    for i, (x, y, w, h) in enumerate(faces):
        print(f"Face {i} found at x: {x}, y: {y}, width: {w}, height: {h}")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"Face {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    # 6. Display the result
    cv2.imshow("Detected Faces", image)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect faces in an image using OpenCV's Haar Cascade Classifier.")
    parser.add_argument("image_path", type=str, nargs='?', default='./meme_faces/bad_luck_brian.png', help="Path to the image file.")
    parser.add_argument("--scale_factor", type=float, default=1.2, help="Scale factor for the detection.")
    parser.add_argument("--min_neighbors", type=int, default=5, help="Minimum neighbors for the detection.")
    parser.add_argument("--min_size", type=tuple, default=(30, 30), help="Minimum size for the detection.")
    args = parser.parse_args()
    detect_faces_in_image(args.image_path, args.scale_factor, args.min_neighbors, args.min_size)
