import matplotlib.pyplot as plt  
import numpy as np 

#landmark angle: 1.8574475597536577 
#robot pose: -3.135
angle = 1.857 
x0 = 0; y0 = 0; 
x1 = 5 * np.cos(angle) 
y1 = 5 * np.sin(angle)  

fig, ax = plt.subplots(figsize=(12,12))  
plt.plot([x0,x1],[y0,y1])  
plt.grid(visible=True)
ax.scatter(x0,y0) 
ax.set_xlim([-7,7]) 
ax.set_ylim([-7,7])
plt.show()