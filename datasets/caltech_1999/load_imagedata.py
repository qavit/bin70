from scipy.io import loadmat
import numpy as np

# Load the .mat file
data = loadmat("./ImageData.mat")

# Extract the SubDir_Data matrix
subdir_data = data["SubDir_Data"]

# Save the matrix as a .npy file
np.save("boundary_boxes.npy", subdir_data)

