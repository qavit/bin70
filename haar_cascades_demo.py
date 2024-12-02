import cv2
import os
import numpy as np


def detect_faces_and_draw(image_path, scale_factor=1.1, min_neighbors=3, min_size=(30, 30)):
    print(f"Processing image: {image_path}")

    # Load the Haar feature classifier XML file
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )

    # Draw green rectangles around detected faces
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"Face {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image


def create_gallery(image_paths, output_path, images_per_row=3):
    images = [detect_faces_and_draw(path) for path in image_paths]
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)

    # Create a blank canvas for the gallery
    num_rows = (len(images) + images_per_row - 1) // images_per_row
    gallery_height = num_rows * max_height
    gallery_width = images_per_row * max_width
    gallery = np.zeros((gallery_height, gallery_width, 3), dtype=np.uint8)

    # Place each image in the gallery without resizing
    for idx, image in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row
        y = row * max_height
        x = col * max_width
        # Calculate padding for centering the image
        y_offset = (max_height - image.shape[0]) // 2
        x_offset = (max_width - image.shape[1]) // 2
        gallery[y + y_offset:y + y_offset + image.shape[0], x + x_offset:x + x_offset + image.shape[1]] = image

    # Save the gallery image
    cv2.imwrite(output_path, gallery)


if __name__ == "__main__":
    image_dir = './meme_faces/'
    output_gallery_path = './demo_by_haar_cascades.jpg'

    # Get all image paths in the directory
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    # Create a gallery of the processed images
    create_gallery(image_paths, output_gallery_path)
    print(f"Gallery created at {output_gallery_path}")

    # Display the gallery image
    gallery_image = cv2.imread(output_gallery_path)
    cv2.imshow("Gallery", gallery_image)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
