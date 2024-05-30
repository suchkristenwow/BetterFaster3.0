import numpy as np
from scipy.spatial.distance import cdist
import cv2 
import pickle 

def compute_hamming_distances(descriptors1, descriptors2):
    # Convert binary arrays to integers for Hamming distance calculation
    d1 = np.packbits(descriptors1, axis=-1)
    d2 = np.packbits(descriptors2, axis=-1)
    
    # Compute Hamming distances
    hamming_distances = cdist(d1, d2, metric='hamming')
    
    # Since Hamming distance from cdist gives the fraction of differing bits,
    # we multiply by the number of bits per descriptor to get actual bit differences
    bit_length = descriptors1.shape[1] * 8  # 32 bytes * 8 bits
    hamming_distances *= bit_length
    
    return hamming_distances

# Load the image
image = cv2.imread('/home/kristen/Downloads/springTree1.jpeg', cv2.IMREAD_GRAYSCALE)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
_, descriptors_0 = orb.detectAndCompute(image, None)

# Load the image
image = cv2.imread('/home/kristen/Downloads/springTree2.jpeg', cv2.IMREAD_GRAYSCALE)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
_, descriptors_1 = orb.detectAndCompute(image, None)

hamming_ds = compute_hamming_distances(descriptors_0,descriptors_1)
min_values = np.min(hamming_ds, axis=1); 
similarity_val = np.mean(min_values)

print("similarity_val: ",similarity_val)