import numpy as np 

lms_i= np.array([[ 20,  10,   5],
 [ 21, -10,   3],
 [ 22, -10,  -5],
 [ 23,  10,  -5],
 [ 24,   6,   3],
 [ 25,  -8,  12],
 [ 26,   0, -11]])  

tmp = lms_i[:,1:]
lm_id_pos = np.array([  0, -11])

i0 = [i for i,x in enumerate(lms_i[:,1]) if x == lm_id_pos[0]]
i1 = [i for i,x in enumerate(lms_i[:,2]) if x == lm_id_pos[1]]
idx = [x for x in i0 if x in i1][0]

print("idx: ",idx)