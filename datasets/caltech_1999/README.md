# Caltech Face Dataset 1999 Manager

This directory contains the Caltech Face Dataset 1999 and a Python script [`caltech_1999_manager.py`](./caltech_1999_manager.py), which automates downloading, organizing, visualizing face images with bounding boxes, and managing dataset files.

## Dataset Overview

The Caltech Face Dataset 1999 consists of 450 frontal face images collected by Markus Weber at the California Institute of Technology. The images are in JPEG format with a resolution of 896 x 592 pixels. The dataset includes approximately 27 unique individuals captured under various lighting conditions, expressions, and backgrounds.

### Metadata

- **ImageData.mat**: A MATLAB file containing the variable `SubDir_Data`, an 8 x 450 matrix. Each column of this matrix holds the coordinates of the face within the image in the form:
  ```
  [x_bot_left y_bot_left x_top_left y_top_left ... x_top_right y_top_right x_bot_right y_bot_right]
  ```

## Script Overview

The `caltech_1999_manager.py` script is designed to manage the dataset by performing the following tasks:

1. **Download and Extract**: Automatically downloads and extracts the dataset if it is not already present.
2. **Organize Images**: Moves images into specified folders based on predefined rules.
3. **Visualize Images**: Creates a collage of randomly selected images with bounding boxes drawn around faces.
4. **Remove Specified Images**: Removes specific images from the dataset as required.
5. **Purge Unwanted Files**: Deletes all files in the directory except `caltech_1999_manager.py`, `rules.json`, and `README.md`, and recursively deletes folders created based on `rules.json`.
6. **Rename README**: Renames the downloaded `README` file to `DS_INFO`.

### Usage

#### Command-Line Options

- `--collage` or `-c`: Create a collage of 9 random images with bounding boxes. If this option is not specified, the collage will not be created.
- `--purge` or `-p`: Purge unwanted files and directories in the dataset directory.

#### Example Commands

```bash
python caltech_1999_manager.py
python caltech_1999_manager.py --collage
python caltech_1999_manager.py --purge
```

### Configuration

- **Rules File**: The `rules.json` file contains the rules for organizing images into folders. Ensure this file is present in the same directory as the script.

### Image Moving Rules

The images are organized into folders based on the rules specified in `rules.json`. These rules are derived from the guidelines provided by [山姆大叔](https://ithelp.ithome.com.tw/articles/10263219).

### Specific Image Removal

The script includes functionality to remove specific images, such as `image_0399.jpg` to `image_0403.jpg`, as per the instructions from [山姆大叔](https://ithelp.ithome.com.tw/articles/10263219).

## Notes

- Ensure that all required Python packages are installed before running the script.
- The script assumes that the dataset URL and file structure remain consistent with the provided example.

## Acknowledgments

- The dataset was collected by Markus Weber at the California Institute of Technology. See more details in [`DS_INFO`](./DS_INFO) and the [data card](https://data.caltech.edu/records/6rjah-hdv18).
- Special thanks to [山姆大叔](https://ithelp.ithome.com.tw/articles/10263219) for providing guidelines on image organization and removal.