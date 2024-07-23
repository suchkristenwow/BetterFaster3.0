import os 
import pickle 
import matplotlib.pyplot as plt 
import numpy as np 

#results_dir = "/media/kristen/easystore1/BetterFaster/kitti_carla_simulator/exp_results"  
results_dir = "/home/kristen/BetterFaster3.1/betterFaster/sim_utils/fake_data/observation_pickles"
n_observations = []
for exp in range(10): 
    n_observations_exp = []
    #with open(os.path.join(results_dir,"reformed_carla_observations/exp"+str(exp)+"reformed_carla_observations.pickle"),"rb") as handle:
    with open(os.path.join(results_dir,"exp"+str(exp)+"observed_cliques.pickle"),"rb") as handle:
        exp_observations = pickle.load(handle)
    for t in range(500): 
        observations_t = exp_observations[t] 
        n_observations.append(len(observations_t)) 
        n_observations_exp.append(len(observations_t)) 
    tmp = []
    c = 0 
    for count in n_observations: 
        if count == 0:
            c += 1  
        else:
            c = 0 
        if c > 0: 
            #print("c: ",c) 
            tmp.append(c) 
            
    print("max(tmp): ",max(tmp)) 

tmp = []
c = 0 
for count in n_observations: 
    if count == 0:
        c += 1  
    else:
        c = 0 
    if c > 0: 
        #print("c: ",c) 
        tmp.append(c) 

print("max(tmp):",max(tmp)) 

plt.plot(np.arange(len(n_observations)),n_observations) 
plt.show() 