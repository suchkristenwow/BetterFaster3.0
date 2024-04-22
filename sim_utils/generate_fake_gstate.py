import numpy as np 
import pickle 

'''
exp0: [0,6]
exp1: [20,26]
exp2: [30,36]
exp3: [41,46]
exp4: [1,53,54,55,56]
exp5: [1,63,64,65,66]
exp6: [1,73,74,75,76]
exp7: [1,83,4,85]
exp8: [1,93,4,95]
exp9: [1,103,4,105]
'''
#tree ids 0,1,2,3
#cone_ids 4,5,6

#T_nu_tree = 3
#T_nu = 10 

'''
tree_ids = [0,1,2,3]
cone_ids = [4,5,6] 

persistence = np.ones((10,7))
persistence[3:,0] = 0 
persistence[4:,1] = 0
persistence[7:,4] = 0

random_array = [10,20,30,40,50,60,70,80,90,100]

seasons: 
winter: 0-2
summer: 3-5
fall: 6-8
winter: 9 
'''
gt_gstates = {} 

#winter 
gt_gstates[0] = np.zeros((7,2)) 
gt_gstates[0][:,0] = np.arange(gt_gstates[0].shape[0])
gt_gstates[0][:,1] = np.ones((7,)) 

gt_gstates[1] = np.zeros((7,2)) 
gt_gstates[1][:,0] = np.array([20,21,22,23,24,25,26]) #ids 
gt_gstates[1][:,1] = np.ones((7,)) 

gt_gstates[2] = np.zeros((7,2)) 
gt_gstates[2][:,0] = np.array([30,31,32,33,34,35,36]) #ids
gt_gstates[2][:,1] = np.ones((7,)) 

#summer 
gt_gstates[3] = np.zeros((6,2)) 
gt_gstates[3][:,0] = np.array([41,42,43,44,45,46])
gt_gstates[3][:,1] = np.array([2,2,2,1,1,1])

gt_gstates[4] = np.zeros((5,2)) 
gt_gstates[4][:,0] = np.array([1,53,54,55,56])
gt_gstates[4][:,1] = np.array([2,2,1,1,1])

gt_gstates[5] = np.zeros((5,2)) 
gt_gstates[5][:,0] = np.array([1,63,64,65,66])
gt_gstates[5][:,1] = np.array([2,2,1,1,1])

#fall 
gt_gstates[6] = np.zeros((5,2)) 
gt_gstates[6][:,0] = np.array([1,73,74,75,76])
gt_gstates[6][:,1] = np.array([3,3,1,1,1]) 

gt_gstates[7] = np.zeros((4,2)) 
gt_gstates[7][:,0] = np.array([1,83,4,85])
gt_gstates[7][:,1] = np.array([3,3,1,1])

gt_gstates[8] = np.zeros((4,2)) 
gt_gstates[8][:,0] = np.array([1,93,4,95])
gt_gstates[8][:,1] = np.array([3,3,1,1])

#winter 
gt_gstates[9] = np.zeros((4,2)) 
gt_gstates[9][:,0] = np.array([1,103,4,105])
gt_gstates[9][:,1] = np.array([1,1,1,1])

with open("./fake_data/gt_gstates.pickle","wb") as handle:
    pickle.dump(gt_gstates,handle)

