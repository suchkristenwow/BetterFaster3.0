import pickle 
import shutil 
import os 
import toml 
import numpy as np 
from utils import simUtils 
import matplotlib.pyplot as plt 

def process_experiment(results_dir,exp,sim_length,all_data_associations,parameters,clique_feat_ids): 
    #print("This is exp: ",exp)
    orig_gt_car_traj = np.genfromtxt(os.path.join(parameters["results_dir"],"gt_car_poses/experiment"+str(exp + 1)+"_gt_car_pose.csv"),delimiter=",")
    gt_car_traj = np.zeros((orig_gt_car_traj.shape[0],3))
    gt_car_traj[:,0] = orig_gt_car_traj[:,0]
    gt_car_traj[:,1] = orig_gt_car_traj[:,1]
    gt_car_traj[:,2] = orig_gt_car_traj[:,5]

    sim_utils = simUtils(exp,all_data_associations,parameters)

    obsd_clique_path = os.path.join(results_dir,"observation_pickles/experiment"+str(exp + 1)+"observed_cliques.pickle")

    with open(obsd_clique_path,"rb") as handle:
        exp_observations = pickle.load(handle)
        print("exp_observations.keys(): ",exp_observations.keys())
        exp_observations = exp_observations[exp]

    if not os.path.exists(os.path.join(results_dir,"reformed_carla_observations")):
        os.mkdir(os.path.join(results_dir,"reformed_carla_observations"))

    with open(os.path.join(results_dir,"reformed_carla_observations/exp"+str(exp)+"reformed_carla_observations.pickle"),"wb") as handle:
        #print("writing: ",os.path.join(results_dir,"reformed_carla_observations/exp"+str(exp)+"reformed_carla_observations.pickle"))
        reformed_observations = {}
        for t in range(sim_length):
            print("this is t: ",t)
            if np.mod(t,10) == 0:
                percent = 100*(t/sim_length) 
                print("{} percent done!".format(np.round(percent,2))) 

            carla_observations_t = exp_observations[t] 
            reform_obs = sim_utils.reform_observations(exp,t,np.array([gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,2]]),carla_observations_t) 
            for obs in reform_obs: 
                if obs['clique_id'] not in clique_feat_ids:
                    print("clique_feat_ids: ",clique_feat_ids)
                    print("obs: ",obs)
                    raise OSError 
            #list of dicts with keys: clique_id,feature_id,feature_des,bearing,range,detection 
            print("reforming observations...")
            reformed_observations[t] = reform_obs 

        pickle.dump(reformed_observations,handle)

    try:
        with open(os.path.join(results_dir,"reformed_carla_observations/exp"+str(exp)+"reformed_carla_observations.pickle"),"rb") as handle:
            this = pickle.load(handle)
    except:
        raise OSError  
    
    return reformed_observations

'''
fig, ax = plt.subplots(figsize=(12,12)) 

sim_length = 1000 

results_dir = "/media/kristen/easystore/BetterFaster/kitti_carla_simulator/exp_results" 

with open("/home/kristen/BetterFaster3.1/configs/carla.toml","r") as f:
    parameters = toml.load(f)

all_data_associations = {}
n_experiments = parameters["experiments"]

for n in range(1,n_experiments+1):
    #experiment1data_association.csv
    data_associations_path = os.path.join(results_dir,"data_association/experiment"+str(n)+"data_association.csv")
    data_associations = np.genfromtxt(data_associations_path,delimiter=" ")
    all_data_associations[n] = data_associations

for exp in range(1,n_experiments):
    orig_gt_car_traj = np.genfromtxt(os.path.join(parameters["results_dir"],"gt_car_poses/experiment"+str(exp + 1)+"_gt_car_pose.csv"),delimiter=",")
    gt_car_traj = np.zeros((orig_gt_car_traj.shape[0],3))
    gt_car_traj[:,0] = orig_gt_car_traj[:,0]
    gt_car_traj[:,1] = orig_gt_car_traj[:,1]
    gt_car_traj[:,2] = orig_gt_car_traj[:,5]

    print("this is exp no:{}...".format(exp))
    clique_feats_path = os.path.join(results_dir,"observation_pickles/experiment"+str(exp + 1)+"all_clique_feats.pickle") 
    with open(clique_feats_path,"rb") as handle: 
        all_clique_feats = pickle.load(handle)
        print("all_clique_feats.keys(): ",all_clique_feats.keys()) 
    exp_observations = process_experiment(exp,sim_length,all_data_associations,parameters,all_clique_feats.keys())   
    for t in exp_observations.keys(): 
        observations_t = exp_observations[t] 
        ax.clear() 
        ax.set_aspect('equal') 
        ax.scatter(gt_car_traj[t,0],gt_car_traj[t,1],color="k",s=5) 
        #plot the gt trees  

        #plot the gt cones 
'''
