import numpy as np
from scipy.spatial.distance import cdist
import cv2 
import pickle 

with open("/home/kristen/BetterFaster3.1/betterFaster/sim_utils/fake_data/gt_gstates.pickle","rb") as handle: 
    data = pickle.load(handle)
    print("data: ",data[6])