import os 
import numpy as np 
import sys 
sys.path.append("../sim_utils/") 
import pickle 
from utils import get_reinitted_id 
import matplotlib.pyplot as plt 
#get_reinitted_id(all_data_associations,n,id_,optional_exp=None):  

results_dir = "/media/kristen/easystore1/BetterFaster/kitti_carla_simulator/exp_results"  
fig, ax = plt.subplots(figsize=(12,12))  

for exp in range(11): 
    data_association_path = os.path.join(results_dir,"data_association/experiment"+str(exp+1)+"data_association.csv") 
    exp_data_association = np.genfromtxt(data_association_path,delimiter=" ")  

    clique_feats_path = os.path.join(results_dir,"observation_pickles/experiment"+str(exp + 1)+"all_clique_feats.pickle")

    with open(clique_feats_path,"rb") as handle:
        all_clique_feats = pickle.load(handle)

    cone_ids_file = os.path.join(results_dir,"cone_ids/experiment"+str(exp + 1)+"cone_ids_.txt")
    # Read the contents of the text file
    with open(cone_ids_file, 'r') as file:
        lines = file.readlines()
    cone_ids = np.unique([int(line.strip()) for line in lines])


    # Convert each line to an integer and store in a list
    tree_ids_file = os.path.join(results_dir,"tree_ids/experiment"+str(exp + 1)+"tree_ids_.txt")
    # Read the contents of the text file
    with open(tree_ids_file, 'r') as file:
        lines = file.readlines()
    # Convert each line to an integer and store in a list
    tree_ids = np.unique([int(line.strip()) for line in lines])  

    gt_trees = [x for x in exp_data_association if x[0] in tree_ids]   
    gt_cones = [x for x in exp_data_association if x[0] in cone_ids] 

    for i in range(len(gt_trees)):
        ax.scatter(gt_trees[i,1],gt_trees[i,2],color='green', marker='*') 
        #plot features 
        

    for i in range(len(gt_cones)): 
        ax.scatter(gt_cones[i,1],gt_cones[i,2],color='orange', marker='^') 

    #plot the observed features and the associated features and pause after each one  
