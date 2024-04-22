import numpy as np 

cone0 = np.ones((10,))
cone1 = np.ones((10,))
cone2 = np.ones((10,)); cone2[7:] = 0
print("cone2: ",cone2)