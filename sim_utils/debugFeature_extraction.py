import pickle 
import os 

#serialize 
'''
print("serialize......")
if not os.path.exists(os.path.join(results_dir,"observation_pickles")):
    os.mkdir(os.path.join(results_dir,"observation_pickles"))

dir_ = os.path.join(results_dir,"observation_pickles")
print("writing: ",os.path.join(dir_,"experiment"+ str(exp +1)+ "observed_cliques.pickle"))
with open(os.path.join(dir_,"experiment"+ str(exp +1)+ "observed_cliques.pickle"),"wb") as handle:
    pickle.dump(obsd_cliques,handle,protocol=pickle.HIGHEST_PROTOCOL)
print("writing: ",os.path.join(dir_,"experiment"+ str(exp +1)+ "all_clique_feats.pickle"))
with open(os.path.join(dir_,"experiment"+ str(exp +1)+ "all_clique_feats.pickle"),"wb") as handle:
    pickle.dump(all_clique_feats,handle,protocol=pickle.HIGHEST_PROTOCOL)
'''

results_dir = "/media/arpg/easystore/BetterFaster/kitti_carla_simulator/exp1_results" 
dir_ = os.path.join(results_dir,"observation_pickles")
exp = 0 

print("opening: ",os.path.join(dir_,"experiment"+ str(exp +1)+ "observed_cliques.pickle"))
with open(os.path.join(dir_,"experiment"+ str(exp +1)+ "observed_cliques.pickle"),"rb") as handle:
    #pickle.dump(obsd_cliques,handle,protocol=pickle.HIGHEST_PROTOCOL)
    obsd_cliques = pickle.load(handle)
    exp_obsd_cliques = obsd_cliques[exp]

with open(os.path.join(dir_,"experiment"+ str(exp +1)+ "all_clique_feats.pickle"),"rb") as handle:
    #pickle.dump(all_clique_feats,handle,protocol=pickle.HIGHEST_PROTOCOL)
    all_clique_feats = pickle.load(handle)

for t in exp_obsd_cliques:
    #print("observations: ",exp_obsd_cliques[t])
    for lm_id in exp_obsd_cliques[t].keys():
        observations = exp_obsd_cliques[t][lm_id]
        if len(observations.keys()) > 1:
            print("there is more than one feature observed for this clique at this timestep:",t)